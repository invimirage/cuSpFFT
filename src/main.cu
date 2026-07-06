#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include "mtx_reader.h"
#include "dense_baseline.h"
#include "sparse_fft.h"
#ifdef HAS_SPFFT
#include <spfft/spfft.hpp>
#endif
#ifdef HAS_CPU_REF
#include "cpu_reference.h"
#endif

#ifdef HAS_SPFFT
// Scatter COO entries (value=1) into a dense row-major float array for SpFFT.
// Must use 64-bit index arithmetic (same rule as coo_to_dense in dense_baseline.cu).
__global__ void spfft_scatter_kernel(const int* row_idx, const int* col_idx,
                                     float* d_dense, int cols, int nnz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nnz)
        d_dense[(size_t)row_idx[i] * cols + col_idx[i]] = 1.0f;
}
#endif

// RAII NVTX range — push on construction, pop on destruction.
struct NvtxRange {
    explicit NvtxRange(const char* label) { nvtxRangePushA(label); }
    ~NvtxRange()                          { nvtxRangePop(); }
};

// Repeats fn N+1 times: 1 warmup (discarded) + N timed (when n_iters > 1),
// or 1 timed run (when n_iters == 1, matching legacy behavior).  Returns the
// LAST result with .ms set to the median of the timed runs.  Optional
// out_min/out_max receive the min/max of the timed runs (untouched if n_iters == 1).
//
// Result must have a `cuFloatComplex* d_output` field that is either nullptr
// or owned by the caller (the helper cudaFree's intermediate results).
template <typename Result, typename Fn>
static Result run_repeated(int n_iters, Fn&& fn,
                            float* out_min = nullptr, float* out_max = nullptr) {
    Result result{};
    std::vector<float> times;
    times.reserve(n_iters);

    if (n_iters > 1) {
        result = fn();
        if (result.d_output) { cudaFree(result.d_output); result.d_output = nullptr; }
    }

    for (int i = 0; i < n_iters; i++) {
        if (i > 0 && result.d_output) { cudaFree(result.d_output); }
        result = fn();
        times.push_back(result.ms);
    }

    if (n_iters > 1) {
        std::sort(times.begin(), times.end());
        result.ms = times[n_iters / 2];
        if (out_min) *out_min = times.front();
        if (out_max) *out_max = times.back();
    }
    return result;
}

// Single-run mode: legacy 2-column output.
// Multi-run mode:  median in column 2, [min/max] + n in trailing brackets.
static void print_bench(const char* label, float median_ms, size_t mem_bytes,
                         int n_iters, float min_ms, float max_ms) {
    if (n_iters > 1) {
        printf("%-26s  %10.3f  %12.2f  [min %.3f, max %.3f]  n=%d\n",
               label, median_ms, mem_bytes / 1e6, min_ms, max_ms, n_iters);
    } else {
        printf("%-26s  %10.3f  %12.2f\n",
               label, median_ms, mem_bytes / 1e6);
    }
}

struct DenseReference {
    bool available = false;
    int rows = 0;
    int cols = 0;
    int freq_cols = 0;
    std::vector<cuFloatComplex> h_freq;
};

struct CheckStats {
    double max_abs = 0.0;
    double max_rel = 0.0;
    double rms_abs = 0.0;
};

static void cuda_check(cudaError_t err) {
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
}

static void write_npy_header(std::ofstream& out, int rows, int cols) {
    const char magic[] = "\x93NUMPY";
    out.write(magic, 6);
    const unsigned char version[2] = {1, 0};
    out.write(reinterpret_cast<const char*>(version), 2);
    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (" +
        std::to_string(rows) + ", " + std::to_string(cols) + "), }";
    const size_t preamble = 10;
    size_t pad = 16 - ((preamble + header.size() + 1) % 16);
    if (pad == 16) pad = 0;
    header.append(pad, ' ');
    header.push_back('\n');
    if (header.size() > 65535) {
        throw std::runtime_error("NumPy header too large");
    }
    const uint16_t header_len = static_cast<uint16_t>(header.size());
    out.put(static_cast<char>(header_len & 0xff));
    out.put(static_cast<char>((header_len >> 8) & 0xff));
    out.write(header.data(), header.size());
}

static void dump_magnitude_npy(const char* path, const SparseFFTResult& result,
                               int rows, int cols) {
    const int freq_cols = cols / 2 + 1;
    const size_t n_freq = (size_t)rows * freq_cols;
    std::vector<cuFloatComplex> host;
    const cuFloatComplex* freq = nullptr;
    if (!result.h_output.empty()) {
        freq = result.h_output.data();
    } else if (result.d_output) {
        host.resize(n_freq);
        cuda_check(cudaMemcpy(host.data(), result.d_output,
                              n_freq * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost));
        freq = host.data();
    } else {
        throw std::runtime_error("no cuSpFFT output available to dump");
    }

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error(std::string("failed to open dump output: ") + path);
    }
    write_npy_header(out, rows, freq_cols);
    std::vector<float> row(freq_cols);
    for (int r = 0; r < rows; ++r) {
        const cuFloatComplex* src = freq + (size_t)r * result.output_cols;
        for (int c = 0; c < freq_cols; ++c) {
            const float x = src[c].x;
            const float y = src[c].y;
            row[c] = std::sqrt(x * x + y * y);
        }
        out.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(float));
    }
}

static DenseReference make_dense_reference(const DenseFFTResult& dense, int rows, int cols) {
    DenseReference ref;
    ref.available = true;
    ref.rows = rows;
    ref.cols = cols;
    ref.freq_cols = cols / 2 + 1;
    ref.h_freq.resize((size_t)rows * ref.freq_cols);

    cuda_check(cudaMemcpy(ref.h_freq.data(), dense.d_output,
                          ref.h_freq.size() * sizeof(cuFloatComplex),
                          cudaMemcpyDeviceToHost));
    return ref;
}

#ifdef HAS_CPU_REF
// Populate the reference from the CPU FFTW3 result.  Used when --cpu-reference
// is set; the CPU output then becomes the ground truth that even dense cuFFT
// and SpFFT are checked against.
static DenseReference make_cpu_reference(CPUReferenceResult&& cpu, int cols) {
    DenseReference ref;
    ref.available = true;
    ref.rows = cpu.rows;
    ref.cols = cols;
    ref.freq_cols = cpu.freq_cols;
    ref.h_freq = std::move(cpu.output);
    return ref;
}
#endif

// Compare a dense d_output against the host reference.  Used for dense cuFFT
// when the reference comes from a CPU FFT.
static CheckStats compare_dense_to_ref(const DenseReference& ref,
                                        const DenseFFTResult& dense) {
    std::vector<cuFloatComplex> h_dense((size_t)ref.rows * ref.freq_cols);
    cuda_check(cudaMemcpy(h_dense.data(), dense.d_output,
                          h_dense.size() * sizeof(cuFloatComplex),
                          cudaMemcpyDeviceToHost));

    CheckStats stats;
    double sum_sq = 0.0;
    const size_t n = h_dense.size();
    for (size_t i = 0; i < n; i++) {
        const double dx = (double)h_dense[i].x - (double)ref.h_freq[i].x;
        const double dy = (double)h_dense[i].y - (double)ref.h_freq[i].y;
        const double abs_err = std::sqrt(dx * dx + dy * dy);
        const double ref_mag = std::sqrt((double)ref.h_freq[i].x * ref.h_freq[i].x
                                       + (double)ref.h_freq[i].y * ref.h_freq[i].y);
        const double rel_err = abs_err / std::max(1.0, ref_mag);
        stats.max_abs = std::max(stats.max_abs, abs_err);
        stats.max_rel = std::max(stats.max_rel, rel_err);
        sum_sq += abs_err * abs_err;
    }
    stats.rms_abs = std::sqrt(sum_sq / (double)n);
    return stats;
}

static void print_dense_check(const char* method, const DenseReference& ref,
                                const DenseFFTResult& dense) {
    if (!ref.available) return;
    CheckStats s = compare_dense_to_ref(ref, dense);
    printf("  check %-19s  max_abs %.3e  max_rel %.3e  rms_abs %.3e\n",
           method, s.max_abs, s.max_rel, s.rms_abs);
}

static CheckStats compare_sparse_to_dense(const DenseReference& ref,
                                          const SparseFFTResult& sparse) {
    if (sparse.output_cols < ref.freq_cols)
        throw std::runtime_error("Sparse output has fewer columns than dense reference");

    std::vector<cuFloatComplex> h_sparse((size_t)ref.rows * ref.freq_cols);
    if (!sparse.h_output.empty()) {
        for (int r = 0; r < ref.rows; r++) {
            std::copy_n(sparse.h_output.data() + (size_t)r * sparse.output_cols,
                        ref.freq_cols,
                        h_sparse.data() + (size_t)r * ref.freq_cols);
        }
    } else {
        cuda_check(cudaMemcpy2D(h_sparse.data(),
                                (size_t)ref.freq_cols * sizeof(cuFloatComplex),
                                sparse.d_output,
                                (size_t)sparse.output_cols * sizeof(cuFloatComplex),
                                (size_t)ref.freq_cols * sizeof(cuFloatComplex),
                                ref.rows,
                                cudaMemcpyDeviceToHost));
    }

    CheckStats stats;
    double sum_sq = 0.0;
    const size_t n = h_sparse.size();
    for (size_t i = 0; i < n; i++) {
        const double dx = (double)h_sparse[i].x - (double)ref.h_freq[i].x;
        const double dy = (double)h_sparse[i].y - (double)ref.h_freq[i].y;
        const double abs_err = std::sqrt(dx * dx + dy * dy);
        const double ref_mag = std::sqrt((double)ref.h_freq[i].x * ref.h_freq[i].x
                                       + (double)ref.h_freq[i].y * ref.h_freq[i].y);
        const double rel_err = abs_err / std::max(1.0, ref_mag);
        stats.max_abs = std::max(stats.max_abs, abs_err);
        stats.max_rel = std::max(stats.max_rel, rel_err);
        sum_sq += abs_err * abs_err;
    }
    stats.rms_abs = std::sqrt(sum_sq / (double)n);
    return stats;
}

static void print_check(const char* method, const DenseReference& ref,
                        const SparseFFTResult& sparse) {
    if (!ref.available)
        return;
    CheckStats s = compare_sparse_to_dense(ref, sparse);
    printf("  check %-19s  max_abs %.3e  max_rel %.3e  rms_abs %.3e\n",
           method, s.max_abs, s.max_rel, s.rms_abs);
}

static void print_usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s <matrix.mtx> [options]\n"
        "Options:\n"
        "  --sparse-only    Skip dense cuFFT baseline\n"
        "  --dense-only     Skip sparse FFT methods\n"
        "  --csc-bluestein  Add CSC variant with custom Bluestein FFT pass 2\n"
        "  --csc-stockham   Use Stockham FFT backend for CSC Bluestein variant\n"
        "  --csc-stockham-smem   Add CSC Stockham variant with smem row FFT (improvement 2)\n"
        "  --csc-stockham-cufft  Add CSC Bluestein variant using cuFFT C2C for pass-2 FFTs\n"
        "  --csc-stockham-stream  Add double-buffered streaming Stockham variant\n"
        "  --csc-stockham-cufft-stream  Add streaming variant using cuFFT C2C backend (minimal device memory)\n"
        "  --csc-stockham-graph   Add CUDA Graph replay Stockham variant\n"
        "  --csc-tile N     Output-row tile size for chunked CSC (default: 1024)\n"
        "  --spfft          SpFFT GPU baseline (requires HAS_SPFFT build)\n"
        "  --csc-binary-lut Experimental binary-CSC byte-mask + LUT pass-1 + cuFFT (streaming)\n"
        "  --csc-binary-stockham-smem  Experimental binary-CSC byte-mask + LUT pass-1 + Stockham smem (non-streaming)\n"
        "  --csc-mixed-radix  Experimental binary-CSC + mixed-radix {2,3} Stockham Bluestein (no cuFFT)\n"
        "  --csc-mixed-radix-graph  Same as --csc-mixed-radix but with CUDA Graph capture/replay\n"
        "  --csc-cufft-smooth  cuFFT Bluestein with fft_len = next_7_smooth(2·cols−1) (best-possible cuFFT baseline)\n"
        "  --cpu-reference  Compute CPU FFT reference (FFTW3, single-threaded); used as ground truth for ALL variants\n"
        "  --repeat N       Run each benchmark N+1 times (1 warmup + N timed); report median, min, max (default 1)\n",
        prog);
}

static void print_separator() {
    printf("%-26s  %10s  %12s\n",
           "Method", "Time (ms)", "Mem (MB)");
    printf("%-26s  %10s  %12s\n",
           "--------------------------", "----------", "------------");
}

int main(int argc, char** argv) {
    if (argc < 2) { print_usage(argv[0]); return 1; }
    if (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
        print_usage(argv[0]);
        return 0;
    }

    bool run_dense   = true;
    bool run_sparse  = true;
    bool run_2pass   = true;
    bool csc_bluestein = false;
    bool csc_stockham = false;
    bool csc_stockham_smem = false;
    bool csc_stockham_cufft = false;
    bool csc_stockham_stream = false;
    bool csc_stockham_cufft_stream = false;
    bool csc_stockham_graph = false;
    bool run_spfft    = false;
    bool csc_binary_lut = false;
    bool csc_binary_stockham_smem = false;
    bool csc_mixed_radix = false;
    bool csc_mixed_radix_graph = false;
    bool csc_cufft_smooth = false;
    bool run_cpu_ref = false;
    std::string dump_mag_path;
    int csc_tile     = 1024;
    int repeat_iters = 1;

    for (int i = 2; i < argc; i++) {
        std::string f(argv[i]);
        if      (f == "--sparse-only") run_dense  = false;
        else if (f == "--dense-only")  { run_sparse = false; run_2pass = false; }
        else if (f == "--csc-bluestein") csc_bluestein = true;
        else if (f == "--csc-stockham") csc_stockham = true;
        else if (f == "--csc-stockham-smem") csc_stockham_smem = true;
        else if (f == "--csc-stockham-cufft") csc_stockham_cufft = true;
        else if (f == "--csc-stockham-stream") csc_stockham_stream = true;
        else if (f == "--csc-stockham-cufft-stream") csc_stockham_cufft_stream = true;
        else if (f == "--csc-stockham-graph") csc_stockham_graph = true;
        else if (f == "--spfft")       run_spfft    = true;
        else if (f == "--csc-binary-lut") csc_binary_lut = true;
        else if (f == "--csc-binary-stockham-smem") csc_binary_stockham_smem = true;
        else if (f == "--csc-mixed-radix") csc_mixed_radix = true;
        else if (f == "--csc-mixed-radix-graph") csc_mixed_radix_graph = true;
        else if (f == "--csc-cufft-smooth") csc_cufft_smooth = true;
        else if (f == "--cpu-reference") run_cpu_ref = true;
        else if (f == "--dump-mag") {
            if (++i >= argc)
                throw std::runtime_error("--dump-mag requires an output .npy path");
            dump_mag_path = argv[i];
        }
        else if (f == "--csc-tile") {
            if (++i >= argc)
                throw std::runtime_error("--csc-tile requires an integer value");
            csc_tile = std::stoi(argv[i]);
            if (csc_tile <= 0)
                throw std::runtime_error("--csc-tile must be positive");
        }
        else if (f == "--repeat") {
            if (++i >= argc)
                throw std::runtime_error("--repeat requires an integer value");
            repeat_iters = std::stoi(argv[i]);
            if (repeat_iters <= 0)
                throw std::runtime_error("--repeat must be positive");
        }
        else {
            throw std::runtime_error("Unknown option: " + f);
        }
    }

    try {
        printf("Matrix: %s\n", argv[1]);
        COOMatrix coo   = read_mtx(argv[1]);
        printf("Size:   %d x %d\n", coo.rows, coo.cols);
        printf("nnz:    %d  (density %.4f%%)\n\n",
               coo.nnz,
               100.0 * coo.nnz / ((double)coo.rows * coo.cols));

        if (!dump_mag_path.empty()) {
            SparseFFTResult sc = sparse_fft_csc_bluestein_cufft_streaming(coo, csc_tile);
            dump_magnitude_npy(dump_mag_path.c_str(), sc, coo.rows, coo.cols);
            printf("Dumped cuSpFFT magnitude: %s  (%d x %d)\n",
                   dump_mag_path.c_str(), coo.rows, coo.cols / 2 + 1);
            return 0;
        }

        print_separator();
        DenseReference dense_ref;

#ifdef HAS_CPU_REF
        // --- CPU FFT reference (FFTW3) ---
        // Runs FIRST so the CPU output becomes the ground truth; every CUDA
        // variant — including dense cuFFT and SpFFT — is then checked against
        // this independent reference.  Useful primarily for sstmodel and
        // benzene; pct20stif is feasible but slow (tens of seconds, ~22 GB host).
        if (run_cpu_ref) {
            NvtxRange _range("CPU reference (FFTW3)");
            try {
                CPUReferenceResult cpu = cpu_fft_r2c_2d(coo);
                printf("%-26s  %10.3f  %12.2f  [CPU, FFTW3 single-thread]\n",
                       "CPU reference (FFTW3)", (float)cpu.ms, cpu.mem_bytes / 1e6);
                dense_ref = make_cpu_reference(std::move(cpu), coo.cols);
            } catch (const std::exception& e) {
                printf("%-26s  %10s  %12s  [%s]\n",
                       "CPU reference (FFTW3)", "FAIL", "FAIL", e.what());
            }
        }
#endif

        // --- Dense cuFFT baseline ---
        if (run_dense) {
            NvtxRange _range("Dense cuFFT (baseline)");
            try {
                // Track whether the reference was already populated by something
                // else (CPU FFT) before we run — controls whether we cross-check
                // dense cuFFT against it.  Comparing dense vs dense is a no-op.
                const bool ref_was_external = dense_ref.available;
                float min_ms = 0, max_ms = 0;
                DenseFFTResult dr = run_repeated<DenseFFTResult>(repeat_iters,
                    [&] { return dense_fft(coo); }, &min_ms, &max_ms);
                if (!dense_ref.available && (run_sparse || run_2pass))
                    dense_ref = make_dense_reference(dr, coo.rows, coo.cols);
                print_bench("Dense cuFFT (baseline)", dr.ms, dr.mem_bytes,
                            repeat_iters, min_ms, max_ms);
                // If the reference came from elsewhere (CPU), check dense cuFFT
                // against it — gives an explicit cuFFT-vs-CPU correctness number.
                if (ref_was_external)
                    print_dense_check("Dense cuFFT", dense_ref, dr);
                cudaFree(dr.d_output);
            } catch (const std::exception& e) {
                printf("%-26s  %10s  %12s  [%s]\n",
                       "Dense cuFFT (baseline)", "OOM", "OOM", e.what());
            cudaGetLastError();  // clear sticky device error for next benchmark
            }
        }

        // ===========================================================================
        // HEADLINE — variant 2 of 4: SpFFT (CSCS sparse-frequency FFT, baseline).
        // Built and linked only when -DENABLE_SPFFT=ON.
        // ===========================================================================
#ifdef HAS_SPFFT
        if (run_spfft) {
            NvtxRange _range("SpFFT (baseline)");
            try {
                const int rows = coo.rows, cols = coo.cols;
                const int n_freq = rows * (cols / 2 + 1);

                // Capture baseline free device memory for runtime peak measurement.
                size_t base_free = 0, _tot = 0;
                cudaMemGetInfo(&base_free, &_tot);

                // Upload COO indices to device
                int *d_row = nullptr, *d_col = nullptr;
                cuda_check(cudaMalloc(&d_row, (size_t)coo.nnz * sizeof(int)));
                cuda_check(cudaMalloc(&d_col, (size_t)coo.nnz * sizeof(int)));
                cuda_check(cudaMemcpy(d_row, coo.row_idx.data(),
                    (size_t)coo.nnz * sizeof(int), cudaMemcpyHostToDevice));
                cuda_check(cudaMemcpy(d_col, coo.col_idx.data(),
                    (size_t)coo.nnz * sizeof(int), cudaMemcpyHostToDevice));

                // Build R2C frequency triplets (x, y, 0): x=0..cols/2, y=0..rows-1
                // y-outer / x-inner → row-major layout matching cuFFT R2C output
                std::vector<int> indices;
                indices.reserve((size_t)n_freq * 3);
                for (int y = 0; y < rows; ++y)
                    for (int x = 0; x <= cols / 2; ++x) {
                        indices.push_back(x);
                        indices.push_back(y);
                        indices.push_back(0);
                    }

                // Create SpFFT Grid (single-precision float, 2D via dimZ=1).
                // maxNumLocalZColumns = n_freq: for dimZ=1, each freq element is its own z-column.
                spfft::GridFloat grid(cols, rows, 1, n_freq, SPFFT_PU_GPU, 0);
                spfft::TransformFloat transform = grid.create_transform(
                    SPFFT_PU_GPU, SPFFT_TRANS_R2C,
                    cols, rows, 1, 1,
                    n_freq, SPFFT_INDEX_TRIPLETS, indices.data());

                // Fill SpFFT's internal GPU space-domain buffer via scatter kernel
                float* d_space = transform.space_domain_data(SPFFT_PU_GPU);
                cuda_check(cudaMemset(d_space, 0, (size_t)rows * cols * sizeof(float)));
                spfft_scatter_kernel<<<(coo.nnz + 255) / 256, 256>>>(
                    d_row, d_col, d_space, cols, coo.nnz);
                cuda_check(cudaGetLastError());

                // Output buffer on device for frequency elements
                cuFloatComplex* d_freq = nullptr;
                cuda_check(cudaMalloc(&d_freq, (size_t)n_freq * sizeof(cuFloatComplex)));

                // Time forward transform N+1 times (1 warmup if N>1, then N timed).
                cudaEvent_t t0, t1;
                cuda_check(cudaEventCreate(&t0));
                cuda_check(cudaEventCreate(&t1));

                if (repeat_iters > 1) {
                    transform.forward(SPFFT_PU_GPU, (float*)d_freq, SPFFT_NO_SCALING);
                    cuda_check(cudaDeviceSynchronize());   // warmup
                }
                std::vector<float> times;
                times.reserve(repeat_iters);
                for (int it = 0; it < repeat_iters; it++) {
                    cuda_check(cudaEventRecord(t0));
                    transform.forward(SPFFT_PU_GPU, (float*)d_freq, SPFFT_NO_SCALING);
                    cuda_check(cudaEventRecord(t1));
                    cuda_check(cudaEventSynchronize(t1));
                    float t = 0.0f;
                    cuda_check(cudaEventElapsedTime(&t, t0, t1));
                    times.push_back(t);
                }
                std::sort(times.begin(), times.end());
                float ms     = times[repeat_iters / 2];
                float min_ms = times.front();
                float max_ms = times.back();

                // Sample free memory at peak — captures SpFFT internal allocations
                // (space-domain buffer, transpose scratch) plus our d_row/d_col/d_freq.
                size_t peak_free = 0;
                cudaMemGetInfo(&peak_free, &_tot);
                size_t mem_bytes = (peak_free < base_free) ? base_free - peak_free : 0;

                print_bench("SpFFT (baseline)", ms, mem_bytes,
                            repeat_iters, min_ms, max_ms);

                // Correctness vs dense_ref
                if (dense_ref.available) {
                    SparseFFTResult fake{};
                    fake.d_output    = d_freq;
                    fake.output_cols = cols / 2 + 1;
                    print_check("SpFFT", dense_ref, fake);
                    fake.d_output = nullptr; // prevent double-free
                }

                cudaFree(d_freq);
                cudaFree(d_row);
                cudaFree(d_col);
                cudaEventDestroy(t0);
                cudaEventDestroy(t1);

            } catch (const std::exception& e) {
                printf("%-26s  %10s  %12s  [%s]\n",
                       "SpFFT (baseline)", "OOM", "OOM", e.what());
                cudaGetLastError();
            }
        }
#endif

        // ===========================================================================
        // HEADLINE — variant 3 of 4: CSC binary-sparse + custom radix-{2,3,9}
        // Stockham mixed-radix Bluestein.  No cuFFT call anywhere.
        // ===========================================================================
        if (run_2pass && csc_mixed_radix) {
            NvtxRange _range("Sparse CSC (mixed-radix)");
            try {
                float min_ms = 0, max_ms = 0;
                SparseFFTResult sc = run_repeated<SparseFFTResult>(repeat_iters,
                    [&] { return sparse_fft_csc_bluestein_mixed_radix(coo, csc_tile); },
                    &min_ms, &max_ms);
                print_bench("Sparse CSC (mixed-radix)", sc.ms, sc.mem_bytes,
                            repeat_iters, min_ms, max_ms);
                print_check("Sparse CSC mixed-radix", dense_ref, sc);
                cudaFree(sc.d_output);
            } catch (const std::exception& e) {
                printf("%-26s  %10s  %12s  [%s]\n",
                       "Sparse CSC (mixed-radix)", "OOM", "OOM", e.what());
                cudaGetLastError();
            }
        }

        // ===========================================================================
        // HEADLINE — variant 4 of 4: CSC binary-sparse + cuFFT 1-D FFT at
        // fft_len = next_7_smooth(2·cols−1).  Best-possible cuFFT data point.
        // ===========================================================================
        if (run_2pass && csc_cufft_smooth) {
            NvtxRange _range("Sparse CSC (cuFFT smooth)");
            try {
                float min_ms = 0, max_ms = 0;
                SparseFFTResult sc = run_repeated<SparseFFTResult>(repeat_iters,
                    [&] { return sparse_fft_csc_bluestein_cufft_smooth(coo, csc_tile); },
                    &min_ms, &max_ms);
                print_bench("Sparse CSC (cufft smooth)", sc.ms, sc.mem_bytes,
                            repeat_iters, min_ms, max_ms);
                print_check("Sparse CSC cufft smooth", dense_ref, sc);
                cudaFree(sc.d_output);
            } catch (const std::exception& e) {
                printf("%-26s  %10s  %12s  [%s]\n",
                       "Sparse CSC (cufft smooth)", "OOM", "OOM", e.what());
                cudaGetLastError();
            }
        }

        // ===========================================================================
        // SPARSE FFT VARIANTS.  Headline path (always runs when its flag is set):
        //   --csc-stockham-smem            Stockham smem row FFT
        //   --csc-stockham-cufft           Bluestein + cuFFT C2C (non-streaming)
        //   --csc-stockham-cufft-stream    Bluestein + cuFFT C2C (streaming — only
        //                                  variant that fits pct20stif on 24 GB)
        //
        // Experimental Bluestein variants live in src/sparse_fft_bluestein.cu and
        // are gated behind their own --csc-* flags.  The earlier direct-DFT,
        // COO 2-pass (custom/cuFFT/GEMM), CSC 2-pass (basic/cached-W/tiled/half-G),
        // CSC streaming and CSR 2-pass paths have been removed; the relevant
        // ablation history is in git (search for sparse_fft_experimental.cu).
        // ===========================================================================

        // // --- Sparse FFT: binary CSC with custom Bluestein FFT pass 2 ---
        // if (run_2pass && csc_bluestein) {
        //     NvtxRange _range(csc_stockham ? "Sparse CSC (Stockham)" : "Sparse CSC (Bluestein)");
        //     try {
        //         SparseFFTResult sb = sparse_fft_csc_bluestein(coo, csc_tile, csc_stockham);
        //         printf("%-26s  %10.3f  %12.2f\n",
        //                csc_stockham ? "Sparse CSC (Stockham)" : "Sparse CSC (Bluestein)",
        //                sb.ms, sb.mem_bytes / 1e6);
        //         print_check(csc_stockham ? "Sparse CSC Stockham" : "Sparse CSC Bluestein",
        //                     dense_ref, sb);
        //         cudaFree(sb.d_output);
        //     } catch (const std::exception& e) {
        //         printf("%-26s  %10s  %12s  [%s]\n",
        //                csc_stockham ? "Sparse CSC (Stockham)" : "Sparse CSC (Bluestein)",
        //                "OOM", "OOM", e.what());
        //         cudaGetLastError();
        //     }
        // }

        // --- Sparse FFT: CSC Stockham with smem row FFT (improvement 2) ---
        if (run_2pass && csc_stockham_smem) {
            NvtxRange _range("Sparse CSC (Stockham smem)");
            try {
                float min_ms = 0, max_ms = 0;
                SparseFFTResult ss = run_repeated<SparseFFTResult>(repeat_iters,
                    [&] { return sparse_fft_csc_bluestein_smem(coo, csc_tile); },
                    &min_ms, &max_ms);
                print_bench("Sparse CSC (Stk smem)", ss.ms, ss.mem_bytes,
                            repeat_iters, min_ms, max_ms);
                print_check("Sparse CSC Stk smem", dense_ref, ss);
                cudaFree(ss.d_output);
            } catch (const std::exception& e) {
                printf("%-26s  %10s  %12s  [%s]\n",
                       "Sparse CSC (Stk smem)", "OOM", "OOM", e.what());
            cudaGetLastError();  // clear sticky device error for next benchmark
            }
        }

        // --- Sparse FFT: CSC Bluestein with cuFFT C2C backend ---
        if (run_2pass && csc_stockham_cufft) {
            NvtxRange _range("Sparse CSC (Bluestein+cuFFT)");
            try {
                float min_ms = 0, max_ms = 0;
                SparseFFTResult sc = run_repeated<SparseFFTResult>(repeat_iters,
                    [&] { return sparse_fft_csc_bluestein_cufft(coo, csc_tile); },
                    &min_ms, &max_ms);
                print_bench("Sparse CSC (Bluestein+cufft)", sc.ms, sc.mem_bytes,
                            repeat_iters, min_ms, max_ms);
                print_check("Sparse CSC Bluestein+cufft", dense_ref, sc);
                cudaFree(sc.d_output);
            } catch (const std::exception& e) {
                printf("%-26s  %10s  %12s  [%s]\n",
                       "Sparse CSC (Bluestein+cufft)", "OOM", "OOM", e.what());
            cudaGetLastError();  // clear sticky device error for next benchmark
            }
        }

        // // --- Sparse FFT: binary CSC with double-buffered streaming Stockham pass 2 ---
        // if (run_2pass && csc_stockham_stream) {
        //     NvtxRange _range("Sparse CSC (Stockham stream)");
        //     try {
        //         SparseFFTResult ss = sparse_fft_csc_stockham_streaming(coo, csc_tile);
        //         printf("%-26s  %10.3f  %12.2f\n",
        //                "Sparse CSC (Stockham stream)", ss.ms, ss.mem_bytes / 1e6);
        //         print_check("Sparse CSC Stk stream", dense_ref, ss);
        //     } catch (const std::exception& e) {
        //         printf("%-26s  %10s  %12s  [%s]\n",
        //                "Sparse CSC (Stockham stream)", "OOM", "OOM", e.what());
        //         cudaGetLastError();
        //     }
        // }

        // --- Sparse FFT: streaming Bluestein with cuFFT C2C backend ---
        if (run_2pass && csc_stockham_cufft_stream) {
            NvtxRange _range("Sparse CSC (Bluestein+cuFFT stream)");
            try {
                float min_ms = 0, max_ms = 0;
                SparseFFTResult sc = run_repeated<SparseFFTResult>(repeat_iters,
                    [&] { return sparse_fft_csc_bluestein_cufft_streaming(coo, csc_tile); },
                    &min_ms, &max_ms);
                print_bench("Sparse CSC (cufft stream)", sc.ms, sc.mem_bytes,
                            repeat_iters, min_ms, max_ms);
                print_check("Sparse CSC cufft stream", dense_ref, sc);
            } catch (const std::exception& e) {
                printf("%-26s  %10s  %12s  [%s]\n",
                       "Sparse CSC (cufft stream)", "OOM", "OOM", e.what());
            cudaGetLastError();  // clear sticky device error for next benchmark
            }
        }

        // --- Sparse FFT: experimental binary CSC + byte-mask LUT pass-1 + cuFFT (streaming) ---
        if (run_2pass && csc_binary_lut) {
            NvtxRange _range("Sparse CSC (binary LUT)");
            try {
                float min_ms = 0, max_ms = 0;
                SparseFFTResult sc = run_repeated<SparseFFTResult>(repeat_iters,
                    [&] { return sparse_fft_csc_bluestein_binary_lut(coo, csc_tile); },
                    &min_ms, &max_ms);
                print_bench("Sparse CSC (binary LUT)", sc.ms, sc.mem_bytes,
                            repeat_iters, min_ms, max_ms);
                print_check("Sparse CSC binary LUT", dense_ref, sc);
            } catch (const std::exception& e) {
                printf("%-26s  %10s  %12s  [%s]\n",
                       "Sparse CSC (binary LUT)", "OOM", "OOM", e.what());
                cudaGetLastError();
            }
        }

        // --- Sparse FFT: mixed-radix Bluestein with CUDA Graph capture/replay ---
        if (run_2pass && csc_mixed_radix_graph) {
            NvtxRange _range("Sparse CSC (mixed-radix graph)");
            try {
                float min_ms = 0, max_ms = 0;
                SparseFFTResult sc = run_repeated<SparseFFTResult>(repeat_iters,
                    [&] { return sparse_fft_csc_bluestein_mixed_radix_graph(coo, csc_tile); },
                    &min_ms, &max_ms);
                print_bench("Sparse CSC (mr-graph)", sc.ms, sc.mem_bytes,
                            repeat_iters, min_ms, max_ms);
                print_check("Sparse CSC mr-graph", dense_ref, sc);
                cudaFree(sc.d_output);
            } catch (const std::exception& e) {
                printf("%-26s  %10s  %12s  [%s]\n",
                       "Sparse CSC (mr-graph)", "OOM", "OOM", e.what());
                cudaGetLastError();
            }
        }

        // --- Sparse FFT: experimental binary CSC + byte-mask LUT + Stockham smem (non-streaming) ---
        if (run_2pass && csc_binary_stockham_smem) {
            NvtxRange _range("Sparse CSC (binary Stk smem)");
            try {
                float min_ms = 0, max_ms = 0;
                SparseFFTResult sc = run_repeated<SparseFFTResult>(repeat_iters,
                    [&] { return sparse_fft_csc_stockham_binary_smem(coo, csc_tile); },
                    &min_ms, &max_ms);
                print_bench("Sparse CSC (binary Stk smem)", sc.ms, sc.mem_bytes,
                            repeat_iters, min_ms, max_ms);
                print_check("Sparse CSC binary Stk smem", dense_ref, sc);
                cudaFree(sc.d_output);
            } catch (const std::exception& e) {
                printf("%-26s  %10s  %12s  [%s]\n",
                       "Sparse CSC (binary Stk smem)", "OOM", "OOM", e.what());
                cudaGetLastError();
            }
        }

        // // --- Sparse FFT: binary CSC with CUDA Graph replay Stockham pass 2 ---
        // if (run_2pass && csc_stockham_graph) {
        //     NvtxRange _range("Sparse CSC (Stockham graph)");
        //     try {
        //         SparseFFTResult sg = sparse_fft_csc_stockham_graph(coo, csc_tile);
        //         printf("%-26s  %10.3f  %12.2f\n",
        //                "Sparse CSC (Stockham graph)", sg.ms, sg.mem_bytes / 1e6);
        //         print_check("Sparse CSC Stk graph", dense_ref, sg);
        //     } catch (const std::exception& e) {
        //         printf("%-26s  %10s  %12s  [%s]\n",
        //                "Sparse CSC (Stockham graph)", "OOM", "OOM", e.what());
        //         cudaGetLastError();
        //     }
        // }

        printf("\n");

    } catch (const std::exception& e) {
        fprintf(stderr, "Fatal error: %s\n", e.what());
        return 1;
    }

    return 0;
}
