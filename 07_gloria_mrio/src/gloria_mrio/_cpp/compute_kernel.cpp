/*
 * GLORIA MRIO C++ Compute Kernel
 *
 * Accelerates the indirect footprint computation:
 *   For each final-supply sector s2 (parallel via OpenMP):
 *     result[:, :, s2, :] = dL_A_Loo[:, cols_s2] @ Y_o_all[cols_s2, :]
 *     reshaped to (n_reg, n_sec, n_fdreg)
 *
 * Build:
 *   pip install pybind11
 *   cd src/gloria_mrio/_cpp && python setup.py build_ext --inplace
 *   cp _compute_kernel*.so ../
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cstring>
#include <cstdint>

#ifdef _OPENMP
#include <omp.h>
#endif

// Use CBLAS if available, otherwise manual matmul
#ifdef USE_CBLAS
extern "C" {
    void cblas_sgemm(int order, int transA, int transB,
                     int M, int N, int K,
                     float alpha, const float* A, int lda,
                     const float* B, int ldb,
                     float beta, float* C, int ldc);
}
#define CblasRowMajor 101
#define CblasNoTrans  111
#endif

namespace py = pybind11;

/*
 * Compute indirect footprint for ONE target sector across ALL FDregs.
 *
 * Args:
 *   dL_A_Loo:     (n_tot, n_o) float32  — precomputed d * L * A_tsec_o * L_oo
 *   Y_o_all:      (n_o, n_fdreg) float32 — final demand for all FDregs
 *   sector_of_o:  (n_o,) int32           — sector assignment of each index_o element
 *   n_reg, n_sec, n_fdreg: dimensions
 *
 * Returns:
 *   result: (n_reg, n_sec, n_sec, n_fdreg) float32
 */
py::array_t<float> compute_indirect_batched(
    py::array_t<float, py::array::c_style | py::array::forcecast> dL_A_Loo_py,
    py::array_t<float, py::array::c_style | py::array::forcecast> Y_o_all_py,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> sector_of_o_py,
    int n_reg, int n_sec, int n_fdreg)
{
    // Get raw pointers
    auto dL_buf = dL_A_Loo_py.request();
    auto Y_buf  = Y_o_all_py.request();
    auto sec_buf = sector_of_o_py.request();

    const float* dL_data   = static_cast<const float*>(dL_buf.ptr);
    const float* Y_data    = static_cast<const float*>(Y_buf.ptr);
    const int32_t* sec_data = static_cast<const int32_t*>(sec_buf.ptr);

    int n_tot = static_cast<int>(dL_buf.shape[0]);
    int n_o   = static_cast<int>(dL_buf.shape[1]);

    // Group columns by sector
    std::vector<std::vector<int>> cols_by_sector(n_sec);
    for (int j = 0; j < n_o; j++) {
        int s = sec_data[j];
        if (s >= 0 && s < n_sec) {
            cols_by_sector[s].push_back(j);
        }
    }

    // Allocate output (zero-initialized)
    auto result = py::array_t<float>({n_reg, n_sec, n_sec, n_fdreg});
    auto res_buf = result.request();
    float* res_data = static_cast<float*>(res_buf.ptr);
    std::memset(res_data, 0, sizeof(float) * n_reg * n_sec * n_sec * n_fdreg);

    // Parallel over final-supply sectors
    #pragma omp parallel for schedule(dynamic)
    for (int s2 = 0; s2 < n_sec; s2++) {
        const auto& cols = cols_by_sector[s2];
        int nc = static_cast<int>(cols.size());
        if (nc == 0) continue;

        // Extract dL columns: dL_sub[i * nc + c] = dL_data[i * n_o + cols[c]]
        std::vector<float> dL_sub(n_tot * nc);
        for (int c = 0; c < nc; c++) {
            int col = cols[c];
            for (int i = 0; i < n_tot; i++) {
                dL_sub[i * nc + c] = dL_data[i * n_o + col];
            }
        }

        // Extract Y rows: Y_sub[c * n_fdreg + k] = Y_data[cols[c] * n_fdreg + k]
        std::vector<float> Y_sub(nc * n_fdreg);
        for (int c = 0; c < nc; c++) {
            int col = cols[c];
            std::memcpy(&Y_sub[c * n_fdreg], &Y_data[col * n_fdreg],
                        sizeof(float) * n_fdreg);
        }

        // Matmul: (n_tot, nc) @ (nc, n_fdreg) -> (n_tot, n_fdreg)
        std::vector<float> chunk(n_tot * n_fdreg, 0.0f);

#ifdef USE_CBLAS
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    n_tot, n_fdreg, nc,
                    1.0f, dL_sub.data(), nc, Y_sub.data(), n_fdreg,
                    0.0f, chunk.data(), n_fdreg);
#else
        // Manual matmul (still fast for small nc ~ 105)
        for (int i = 0; i < n_tot; i++) {
            for (int c = 0; c < nc; c++) {
                float dL_val = dL_sub[i * nc + c];
                if (dL_val == 0.0f) continue;
                for (int k = 0; k < n_fdreg; k++) {
                    chunk[i * n_fdreg + k] += dL_val * Y_sub[c * n_fdreg + k];
                }
            }
        }
#endif

        // Reshape and store: chunk[i, k] -> result[i/n_sec, i%n_sec, s2, k]
        for (int i = 0; i < n_tot; i++) {
            int reg = i / n_sec;
            int sec = i % n_sec;
            // result index: reg * (n_sec * n_sec * n_fdreg) + sec * (n_sec * n_fdreg) + s2 * n_fdreg + k
            int base_out = reg * (n_sec * n_sec * n_fdreg) + sec * (n_sec * n_fdreg) + s2 * n_fdreg;
            int base_in  = i * n_fdreg;
            std::memcpy(&res_data[base_out], &chunk[base_in], sizeof(float) * n_fdreg);
        }
    }

    return result;
}


PYBIND11_MODULE(_compute_kernel, m) {
    m.doc() = "GLORIA MRIO C++ compute kernel (OpenMP parallel)";
    m.def("compute_indirect_batched", &compute_indirect_batched,
          "Compute indirect footprint for one target sector across all FDregs.\n\n"
          "Args:\n"
          "  dL_A_Loo: (n_tot, n_o) float32\n"
          "  Y_o_all: (n_o, n_fdreg) float32\n"
          "  sector_of_o: (n_o,) int32\n"
          "  n_reg, n_sec, n_fdreg: int\n\n"
          "Returns: (n_reg, n_sec, n_sec, n_fdreg) float32",
          py::arg("dL_A_Loo"), py::arg("Y_o_all"),
          py::arg("sector_of_o"),
          py::arg("n_reg"), py::arg("n_sec"), py::arg("n_fdreg"));
}
