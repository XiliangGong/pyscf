#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include "../np_helper/np_helper.h"
#include "./mp2.h"
#include "../config.h"
#include "time.h"

// 使用 Accelerate 框架
#include <Accelerate/Accelerate.h>

/* Generate 3D array row pointers */
const double ***gen_ptr_arr_3d(const double *p0, const size_t n, const size_t m, const size_t l) {
    size_t i, j;
    const double *p;
    const double ***parr = malloc(sizeof(double **) * n);
    for (i = 0, p = p0; i < n; ++i) {
        parr[i] = malloc(sizeof(double *) * m);
        for (j = 0; j < m; ++j) {
            parr[i][j] = p;
            p += l;
        }
    }
    return parr;
}

/* Elementwise multiply using vDSP_vmulD */
void elementwise_multiply(int size, const double *array1, const double *array2, double *result) {
    vDSP_vmulD(array1, 1, array2, 1, result, 1, size);
}

/* Expand tau_cd array */
double ***expand_tau_cd(const double *tau_cd, int naux, int nocci_nvir) {
    double ***expanded_tau_cd = malloc(naux * sizeof(double **));
    for (int i = 0; i < naux; i++) {
        expanded_tau_cd[i] = malloc(nocci_nvir * sizeof(double *));
        for (int j = 0; j < nocci_nvir; j++) {
            expanded_tau_cd[i][j] = malloc(sizeof(double));
            expanded_tau_cd[i][j][0] = tau_cd[i * nocci_nvir + j];
        }
    }
    return expanded_tau_cd;
}

/* Generate row pointers for a 2D array of size n-by-m */
const double **_gen_ptr_arr(const double *p0, const size_t n, const size_t m) {
    size_t i;
    const double *p;
    const double **parr = malloc(sizeof(double *) * n);
    for (i = 0, p = p0; i < n; ++i) {
        parr[i] = p;
        p += m;
    }
    return parr;
}

// Generate RPA jobs without symmetry
size_t _RPA_gen_jobs(CacheJob *jobs, const size_t i0, const size_t j0, const size_t nocci, const size_t noccj) {
    size_t i, j, ii, jj, m;
    for (m = 0, i = 0, ii = i0; i < nocci; ++i, ++ii) {
        for (j = 0, jj = j0; j < noccj; ++j, ++jj) {
            jobs[m].i = i;
            jobs[m].j = j;
            ++m;
        }
    }
    return m;
}

/* Compute SOSEX energy without OpenMP */
void energy_sosex(double *ed_out, double *ex_out, const double *u2, const double *iJ, const double *jJ, 
                  const int i0, const int j0, const int nocci, const int noccj, const double *tau_cd, 
                  int nocc, int nvir, int naux, int nauxv, double max_memory) {
    printf("Running updated energy_sosex function - Accelerate version\n");
    double ed = 0.0;
    double ex = 0.0;

    const double D0 = 0.0;
    const double D1 = 1.0;

    size_t dsize = sizeof(double);
    size_t nvv = nvir * nvir;
    size_t nvxv = nvir * nauxv;
    size_t mem_avail = max_memory * 1e6;

    CacheJob *jobs = malloc(sizeof(CacheJob) * nocci * noccj);
    size_t njob = _RPA_gen_jobs(jobs, i0, j0, nocci, noccj);

    const double **parr_iaJ = _gen_ptr_arr(iJ, nocci, nvxv);
    const double **parr_jbJ = _gen_ptr_arr(jJ, noccj, nvxv);
    const double **parr_u2 = _gen_ptr_arr(u2, nocci, nvir * nauxv);
    const double ***parr_tau_cd = gen_ptr_arr_3d(tau_cd, naux, nocci, nvir);

    double *cache = malloc(dsize * nvv * 3);
    double *vab = cache;
    double *t2ab = vab + nvv;
    double *t2abT = t2ab + nvv;

    for (size_t m = 0; m < njob; ++m) {
        size_t i = jobs[m].i;
        size_t j = jobs[m].j;

        const double *iaJ = parr_iaJ[i];
        const double *jbJ = parr_jbJ[j];

        // 使用 cblas_dgemm 替代 fblas 的 dgemm_
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    nvir, nvir, nauxv, D1, jbJ, nauxv, iaJ, nauxv, D0, vab, nvir);

        double *t2ab_total = calloc(nvir * nvir, dsize);
        for (int w = 0; w < naux; ++w) {
            double *u2w_i = malloc(nvxv * dsize);
            double *u2w_j = malloc(nvxv * dsize);

            elementwise_multiply(nvxv, parr_u2[i], (double *)parr_tau_cd[w][i], u2w_i);
            elementwise_multiply(nvxv, parr_u2[j], (double *)parr_tau_cd[w][j], u2w_j);

            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        nvir, nvir, nauxv, D1, u2w_j, nauxv, u2w_i, nauxv, D0, t2ab, nvir);

            for (int c = 0; c < nvir * nvir; ++c) {
                t2ab_total[c] -= t2ab[c];
            }

            free(u2w_i);
            free(u2w_j);
        }

        for (int c = 0; c < nvir; ++c) {
            for (int d = 0; d < nvir; ++d) {
                t2abT[d * nvir + c] = t2ab_total[c * nvir + d];
            }
        }

        ed += cblas_ddot(nvir * nvir, vab, 1, t2ab_total, 1) * 2;
        ex -= cblas_ddot(nvir * nvir, vab, 1, t2abT, 1);

        free(t2ab_total);
    }

    *ed_out = ed;
    *ex_out = ex;

    free(cache);
    free(jobs);
}
