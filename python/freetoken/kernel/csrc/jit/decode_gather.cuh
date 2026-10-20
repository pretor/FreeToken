#include <freetoken/tensor.h>
#include <freetoken/utils.cuh>
#include <freetoken/utils.h>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstddef>
#include <cstdint>

// ---------------------------------------------------------------------------
// Decode expert gather over TMA bulk copies (opt-in, QWEN36_DECODE_GATHER=tuned; see
// kernel/decode_gather.py).
//
// Contract: identical to fast_index_copy_multi (fast_index_copy.cuh). For every bank b and every
// i < num_indices, copy row src_indices[i] of the bank at src_ptrs[b] to row dst_indices[i] of
// the bank at dst_ptrs[b]; bank b's rows are feat_bytes[b] bytes (a multiple of 16, 16-byte
// aligned bases). The miss count is read on the device, so the launch is fixed-shape and
// CUDA-graph capturable.
//
// Why: fast_index_copy_multi gives every bank the same CTAs (the 1 MiB gate_up row of an NVFP4
// expert gets 1/6 of the threads, the 4 KiB global rows idle) and keeps one 16-B load per thread
// in flight, 48 CTAs x 1024 threads (~768 KiB), so its rate depends on how well the PCIe/IOMMU
// path absorbs that many small reads of mapped host memory (4 KiB host pages or a busy host fall
// to ~17-25 GB/s). Here the call's TOTAL missed bytes form one work list of chunks (each inside
// one bank row) handed out round-robin to a few CTAs, and each CTA streams its chunks with
// cp.async.bulk through a small shared-memory ring.
// ---------------------------------------------------------------------------

namespace qwen_decode_gather {

inline constexpr int kMaxBanks = 8;

struct Params {
    const int64_t* __restrict__ dst_ptrs;      // [B] device, slot-cache base address per bank
    const int64_t* __restrict__ src_ptrs;      // [B] device, GPU-visible host-bank base address per bank
    const int64_t* __restrict__ feat_bytes;    // [B] device, per-row bytes (multiple of 16)
    const void* __restrict__ dst_indices;      // [L]
    const void* __restrict__ src_indices;      // [L]
    const int64_t* __restrict__ valid_length;  // [1] or null
    int64_t length;                            // L (the miss count is clamped to it)
    int num_banks;                             // B <= kMaxBanks
};

// Streaming 16-B load / plain 16-B store for the pre-sm_90 fallback (LDG.E.NA.128.CONSTANT / STG.E.128).
__device__ __forceinline__ uint4 load_stream(const uint4* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

__device__ __forceinline__ void store_plain(uint4* p, const uint4& v) {
    asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
                 : "memory");
}

// ---------------------------------------------------------------------------
// Kernel (sm_90+): the call's rows x banks, cut into chunks of up to kChunk bytes (each inside
// one bank row), handed out round-robin to the CTAs. One thread per CTA runs a
// kStages-deep ring of shared-memory buffers: cp.async.bulk global(mapped host)->shared,
// completion on an mbarrier, then cp.async.bulk shared->global into the slot. Bytes in
// flight = kBlocks * kStages * kChunk, issued by the SMs' bulk-copy units, so a few CTAs keep
// the PCIe read queue full. Pre-sm_90 builds fall back to a warp LDG copy of the same chunks
// (the module must still compile for every arch of a multi-arch kernel-cache build).
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ uint64_t global_addr(const void* p) {
    uint64_t g;
    asm volatile("cvta.to.global.u64 %0, %1;" : "=l"(g) : "l"(p));
    return g;
}

template <typename IdType, int kStages, int kChunk>
__global__ __launch_bounds__(32) void decode_gather_tma_kernel(const __grid_constant__ Params p) {
    static_assert(kStages >= 1 && kChunk >= 16 && kChunk % 16 == 0);
    extern __shared__ __align__(128) uint8_t ring[];  // kStages * kChunk bytes (dynamic)
    __shared__ __align__(8) uint64_t bars[kStages];
    __shared__ int64_t s_src[kMaxBanks];
    __shared__ int64_t s_dst[kMaxBanks];
    __shared__ int64_t s_feat[kMaxBanks];
    __shared__ int64_t s_stage_dst[kStages];
    __shared__ uint32_t s_stage_bytes[kStages];

    const int nb = p.num_banks < kMaxBanks ? p.num_banks : kMaxBanks;
    const int lane = static_cast<int>(threadIdx.x);
    if (lane < nb) {
        s_src[lane] = p.src_ptrs[lane];
        s_dst[lane] = p.dst_ptrs[lane];
        s_feat[lane] = p.feat_bytes[lane];
    }
    int64_t n = p.valid_length ? p.valid_length[0] : p.length;
    n = n < p.length ? n : p.length;
    if (n <= 0 || nb <= 0) {
        return;
    }
    __syncwarp();

    uint32_t first[kMaxBanks];
    uint32_t per_row = 0;
#pragma unroll
    for (int k = 0; k < kMaxBanks; ++k) {
        first[k] = k < nb ? per_row : 0xFFFFFFFFu;
        if (k < nb) {
            per_row += static_cast<uint32_t>((s_feat[k] + kChunk - 1) / kChunk);
        }
    }
    if (per_row == 0) {
        return;
    }
    const auto* di = static_cast<const IdType*>(p.dst_indices);
    const auto* si = static_cast<const IdType*>(p.src_indices);
    const uint32_t rows = static_cast<uint32_t>(n);

    // chunk c = row * per_row + q for c = blockIdx.x, blockIdx.x + gridDim.x, ...
    uint32_t row = blockIdx.x / per_row;
    uint32_t q = blockIdx.x - row * per_row;
    const uint32_t step_row = gridDim.x / per_row;
    const uint32_t step_q = gridDim.x - step_row * per_row;
    auto locate = [&](const uint8_t*& src, uint8_t*& dst, uint32_t& bytes) {
        int b = 0;
        uint32_t fb = 0;
#pragma unroll
        for (int k = 1; k < kMaxBanks; ++k) {
            if (q >= first[k]) {
                b = k;
                fb = first[k];
            }
        }
        const int64_t feat = s_feat[b];
        const int64_t off = static_cast<int64_t>(q - fb) * kChunk;
        src = reinterpret_cast<const uint8_t*>(s_src[b]) + static_cast<int64_t>(si[row]) * feat + off;
        dst = reinterpret_cast<uint8_t*>(s_dst[b]) + static_cast<int64_t>(di[row]) * feat + off;
        const int64_t rem = feat - off;
        bytes = static_cast<uint32_t>(rem < kChunk ? rem : kChunk);
        q += step_q;
        row += step_row;
        if (q >= per_row) {
            q -= per_row;
            ++row;
        }
    };

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    if (lane != 0) {
        return;
    }
#pragma unroll
    for (int s = 0; s < kStages; ++s) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(smem_addr(&bars[s])) : "memory");
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    auto issue = [&](int s) {
        const uint8_t* src;
        uint8_t* dst;
        uint32_t bytes;
        locate(src, dst, bytes);
        s_stage_dst[s] = reinterpret_cast<int64_t>(dst);
        s_stage_bytes[s] = bytes;
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_addr(&bars[s])), "r"(bytes)
                     : "memory");
        // Destination state space .shared::cta, NOT .shared::cluster: on sm_120 ptxas lowers the
        // .shared::cluster form to a CALL into the driver syscall __cuda_syscall_cp_async_bulk_unicast,
        // and a kernel containing a syscall makes the driver grow the context's per-thread stack
        // (cudaLimitStackSize 1024 -> 14416 B) for every resident thread on its first launch:
        // ~1.37 GiB of device memory on a 70-SM part, never returned. The .shared::cta form
        // (PTX 8.6+) is a native UBLKCP, same bytes, same speed, no stack growth.
        asm volatile(
            "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                smem_addr(ring + static_cast<size_t>(s) * kChunk)),
            "l"(global_addr(src)), "r"(bytes), "r"(smem_addr(&bars[s]))
            : "memory");
    };
    int inflight = 0;
#pragma unroll
    for (int s = 0; s < kStages; ++s) {
        if (row < rows) {
            issue(s);
            ++inflight;
        }
    }
    uint32_t parity = 0;
    int head = 0;
    while (inflight > 0) {
        uint32_t done = 0;
        while (!done) {
            asm volatile(
                "{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\tselp.u32 %0, 1, 0, p;\n\t}"
                : "=r"(done)
                : "r"(smem_addr(&bars[head])), "r"((parity >> head) & 1u)
                : "memory");
        }
        parity ^= 1u << head;
        // order the observed TMA write of this buffer before the bulk store's read of it
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(
                         global_addr(reinterpret_cast<void*>(s_stage_dst[head]))),
                     "r"(smem_addr(ring + static_cast<size_t>(head) * kChunk)), "r"(s_stage_bytes[head])
                     : "memory");
        asm volatile("cp.async.bulk.commit_group;" ::: "memory");
        --inflight;
        if (row < rows) {
            // the bulk store just issued must finish reading this buffer before it is refilled
            asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
            issue(head);
            ++inflight;
        }
        head = head + 1 == kStages ? 0 : head + 1;
    }
    asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
#else
    // pre-sm_90: the warp copies its chunks with 16-B loads/stores
    while (row < rows) {
        const uint8_t* src;
        uint8_t* dst;
        uint32_t bytes;
        locate(src, dst, bytes);
        for (uint32_t i = static_cast<uint32_t>(lane) * 16; i < bytes; i += 32 * 16) {
            store_plain(reinterpret_cast<uint4*>(dst + i), load_stream(reinterpret_cast<const uint4*>(src + i)));
        }
    }
#endif
}

}  // namespace qwen_decode_gather

// TMA variant: kBlocks CTAs x 32 threads, kStages * kChunk bytes of dynamic shared memory.
template <int kBlocks, int kStages, int kChunk>
struct DecodeGatherTmaKernel {
    static constexpr int kSmem = kStages * kChunk;

    static void run(
        tvm::ffi::TensorView dst_ptrs,
        tvm::ffi::TensorView src_ptrs,
        tvm::ffi::TensorView feat_bytes,
        tvm::ffi::TensorView dst_indices,
        tvm::ffi::TensorView src_indices,
        tvm::ffi::Optional<tvm::ffi::TensorView> num_indices
    ) {
        using namespace host;
        auto device = SymbolicDevice{};
        auto B = SymbolicSize{"num_banks"};
        auto L = SymbolicSize{"indices length"};
        auto ptr_dtype = SymbolicDType{};
        auto indices_dtype = SymbolicDType{};
        auto num_indices_dtype = SymbolicDType{};

        TensorMatcher({B}).with_dtype<int64_t>(ptr_dtype).with_device<kDLCUDA>(device)
            .verify(dst_ptrs).verify(src_ptrs).verify(feat_bytes);
        TensorMatcher({L}).with_dtype<int32_t, int64_t>(indices_dtype).with_device<kDLCUDA>(device)
            .verify(dst_indices).verify(src_indices);

        const int64_t* valid_length = nullptr;
        if (num_indices.has_value()) {
            TensorMatcher({1}).with_dtype<int64_t>(num_indices_dtype).with_device<kDLCUDA>(device)
                .verify(num_indices.value());
            valid_length = static_cast<const int64_t*>(num_indices.value().data_ptr());
        }

        const int num_banks = static_cast<int>(B.unwrap());
        const auto length = static_cast<int64_t>(L.unwrap());
        RuntimeCheck(num_banks >= 1 && num_banks <= qwen_decode_gather::kMaxBanks,
                     "decode_gather_tma: 1..8 banks supported, got ", num_banks);
        RuntimeCheck(length < (int64_t{1} << 31), "decode_gather_tma: indices length must be < 2^31");
        const auto params = qwen_decode_gather::Params{
            static_cast<const int64_t*>(dst_ptrs.data_ptr()),
            static_cast<const int64_t*>(src_ptrs.data_ptr()),
            static_cast<const int64_t*>(feat_bytes.data_ptr()),
            dst_indices.data_ptr(),
            src_indices.data_ptr(),
            valid_length,
            length,
            num_banks,
        };
        const auto use_int32 = indices_dtype.unwrap().bits == 32;
        const auto kernel = use_int32
            ? qwen_decode_gather::decode_gather_tma_kernel<int32_t, kStages, kChunk>
            : qwen_decode_gather::decode_gather_tma_kernel<int64_t, kStages, kChunk>;
        // Opt in to kSmem bytes of dynamic shared memory before every launch (host-side function
        // attribute, not a stream operation, so it is legal under graph capture and costs nothing
        // at replay). Needed even at exactly 48 KiB: the static mbarrier/descriptor arrays push
        // the block past the 48 KiB no-opt-in limit.
        RuntimeCheck(::cudaFuncSetAttribute(kernel, ::cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem) == ::cudaSuccess,
                     "decode_gather_tma: cannot reserve ", kSmem, " bytes of shared memory");
        LaunchKernel(static_cast<std::size_t>(kBlocks), 32, device.unwrap(), static_cast<std::size_t>(kSmem))(
            kernel, params);
    }
};
