#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include "../np_helper/np_helper.h"
#include "./mp2.h"
#include "../config.h"
#include "time.h"
#include "../vhf/fblas.h"
// #include <immintrin.h> 

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



void elementwise_multiply(int size, const double *array1, const double *array2, double *result) {
    // #pragma omp parallel for
    for (int k = 0; k < size; k++) {
        result[k] = array1[k] * array2[k];
    }
    // vDSP_vmulD(array1, 1, array2, 1, result, 1, size);


    // vdmul_openblas(size, array1, array2, result);

 

}
// Expand tau_cd array
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

// Generate row pointers for a 2D array of size n-by-m
const double ** _gen_ptr_arr(const double *p0, const size_t n, const size_t m) {
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


// Compute SOSEX energy without OpenMP

void energy_sosex(double *ed_out, double *ex_out, const double *u2, const double *iJ, const double *jJ, 
                  const int i0, const int j0, const int nocci, const int noccj, const double *tau_cd, 
                  int nocc, int nvir, int naux, int nauxv, double max_memory) {
    printf("Running updated energy_sosex function - version 2\n");
    double ed = 0.0;
    double ex = 0.0;

    const double D0 = 0;
    const double D1 = 1;
    const int I1 = 1;

    size_t dsize = sizeof(double);
    size_t nvv = nvir * nvir;
    size_t nvxv = nvir * nauxv;
    size_t mem_avail = max_memory * 1e6;
    int occblksize = fmax(1, fmin(nocc, (int)sqrt(mem_avail / (dsize * nvv * 3))));

    time_t start1 = time(NULL);
    CacheJob *jobs = malloc(sizeof(CacheJob) * nocci * noccj);
    size_t njob = _RPA_gen_jobs(jobs, i0, j0, nocci, noccj);    
    // time_t end1 = time(NULL);
    // printf(" time: %.6f seconds\n", (double)(end1 - start1));



    // clock_t start2 = clock();
    const double **parr_iaJ = _gen_ptr_arr(iJ, nocci, nvxv);
    const double **parr_jbJ = _gen_ptr_arr(jJ, noccj, nvxv);
    const double **parr_u2 = _gen_ptr_arr(u2, nocci, nvir * nauxv);
    const double ***parr_tau_cd = gen_ptr_arr_3d(tau_cd, naux, nocci, nvir);    
    // clock_t end2 = clock();
    // printf(" time: %.6f seconds\n", (double)(end2 - start2) / CLOCKS_PER_SEC);



    clock_t start3 = clock();
    double ***expanded_tau_cd = malloc(naux * sizeof(double **));
    for (int i = 0; i < naux; ++i) {
        expanded_tau_cd[i] = malloc(nocci * sizeof(double *));
        for (int j = 0; j < nocci; ++j) {
            expanded_tau_cd[i][j] = (double *)malloc(nvir * nauxv * sizeof(double));
            for (int k = 0; k < nvir; ++k) {
                for (int l = 0; l < nauxv; ++l) {
                    expanded_tau_cd[i][j][k * nauxv + l] = parr_tau_cd[i][j][k];
                }
            }
        }
    }    
    clock_t end3 = clock();
    // printf(" time: %.6f seconds\n", (double)(end3 - start3) / CLOCKS_PER_SEC);


    // #pragma omp parallel
    clock_t start4 = clock();
    double *cache = malloc(dsize * nvv * 3);
    if (cache == NULL) {
        fprintf(stderr, "Memory allocation failed for cache.\n");
        exit(EXIT_FAILURE);
    }

    double *vab = cache;
    double *t2ab = vab + nvv;
    double *t2abT = t2ab + nvv;    
    clock_t end4 = clock();
    // printf(" time: %.6f seconds\n", (double)(end4 - start4) / CLOCKS_PER_SEC);


    // 定义一些变量来记录总耗时
    double total_dgemm_time = 0.0;
    double total_iterate_w_time = 0.0;
    double total_accumulate_time = 0.0;
    double em1_total = 0.0;
    double em2_total = 0.0;
    // #pragma omp for schedule(dynamic, 4)
    time_t start5 = time(NULL);
    for (size_t m = 0; m < njob; ++m) {

        size_t i = jobs[m].i;
        size_t j = jobs[m].j;
        // printf("i = %zu, j = %zu", i ,j );

        const double *iaJ = parr_iaJ[i];
        const double *jbJ = parr_jbJ[j];
        time_t s1 = time(NULL);
        //jaJ is a 1D pointor but logically a 2D array
        dgemm_("T", "N", &nvir, &nvir, &nauxv, &D1, jbJ, &nauxv, iaJ, &nauxv, &D0, vab, &nvir);
        time_t e1 = time(NULL);
        double dgemm_time = difftime(e1, s1);
        // printf("fdgemm time for m= %zu: %.15f seconds\n", m, dgemm_time);
        total_dgemm_time += dgemm_time;

        double **u2w_i = malloc(naux * sizeof(double *));
        double **u2w_j = malloc(naux * sizeof(double *));
        if (u2w_i == NULL) {
            fprintf(stderr, "Memory allocation failed for u2w_i.\n");
            exit(EXIT_FAILURE);
        }
        // double *u2w_i = malloc(nvxv * sizeof(double));
        // double *u2w_j = malloc(nvxv * sizeof(double));
        // 为每个 u2w_i[w] 和 u2w_j[w] 分配 nvxv 的内存

        const double *u2_i = parr_u2[i];
        const double *u2_j = parr_u2[j];

        double *t2ab_total = malloc(nvir * nvir * sizeof(double));
        for (int uuu = 0; uuu < nvir * nvir; uuu++) {
            t2ab_total[uuu] = 0.0;
        }



        time_t s2 = time(NULL);
        int nvxv_int = (int)nvxv;
        for (int w = 0; w < naux; ++w) {
            u2w_i[w] = malloc(nvxv * sizeof(double));
            u2w_j[w] = malloc(nvxv * sizeof(double));
            // vdMul(nvxv_int, u2_i, (double *)expanded_tau_cd[w][i], u2w_i[w]  );
            // vdMul(nvxv_int, u2_j, (double *)expanded_tau_cd[w][j], u2w_j[w]  );

            // iu2w = tw[i0:i1,:,None]*u2[i0:i1]  （nocc, nvir, nauxv）
            time_t s22 = time(NULL);
            elementwise_multiply(nvxv_int, u2_i, (double *)expanded_tau_cd[w][i], u2w_i[w]);
            elementwise_multiply(nvxv_int, u2_j, (double *)expanded_tau_cd[w][j], u2w_j[w]);
            time_t e22 = time(NULL);
            double em1 = difftime(e22, s22);
            em1_total += em1;


            double *t2ab = malloc(nvv * sizeof(double));
            for (int i = 0; i < nvir * nvir; i++) {
                t2ab[i] = 0.0;
            }

            time_t s23 = time(NULL);
            dgemm_("T", "N", &nvir, &nvir, &nauxv, &D1, u2w_j[w], &nauxv, u2w_i[w], &nauxv, &D0, t2ab, &nvir);
            time_t e23 = time(NULL);
            double em2 = difftime(e23, s23);
            em2_total += em2;


            for (int cc = 0; cc < nvir * nvir; cc++) {
                t2ab_total[cc] -= t2ab[cc];
            }
            // printf("first 10 of t2ab:\n");
            // for (int aa = 0; aa < 10; ++aa){
            //   printf("t2ab[%d] = %.15e\n  ", aa, t2ab[aa]);
            // }
            free(t2ab);
        }
        time_t e2 = time(NULL);
        double iterate_w_time = difftime(e2, s2);
        // printf("Iterate w time for m = %zu: %.15f seconds\n", m, iterate_w_time);
        total_iterate_w_time += iterate_w_time;

        int nvv_int = (int)nvv;
        for (int c = 0; c < nvir; c++) {
            for (int d = 0; d < nvir; d++) {
                t2abT[d * nvir + c] = t2ab_total[c * nvir + d];
            }
        }

        time_t s3 = time(NULL);
        ed += ddot_(&nvv_int, vab, &I1, t2ab_total, &I1) * 2;
        ex -= ddot_(&nvv_int, vab, &I1, t2abT, &I1);
        // ed += cblas_ddot(nvv_int, vab, I1, t2ab_total, I1) * 2;
        // ex -= ddot_(nvv_int, vab, I1, t2abT, I1);
        time_t e3 = time(NULL);
        double accumulate_time = difftime(e3, s3);
        // printf("accumulate ed+ex time for m=%zu: %.15f seconds\n", m, accumulate_time);
        total_accumulate_time += accumulate_time;

        free(u2w_i);
        free(u2w_j);
        free(t2ab_total);
    }
    // 打印每个部分的总耗时
    printf("Total dgemm time: %.6f seconds\n", total_dgemm_time);
    printf("Total iterate w time: %.6f seconds\n", total_iterate_w_time);
    printf("Total accumulate time: %.10f seconds\n", total_accumulate_time);
    printf("Total em1 time: %.10f seconds\n", em1_total);
    printf("Total em2 time: %.10f seconds\n", em2_total);
    
    time_t end5 = time(NULL);
    printf(" time: %.6f seconds\n", difftime(end5, start1));

    // #pragma omp critical
    clock_t start6 = clock();
    *ed_out = ed;
    *ex_out = ex;    
    clock_t end6 = clock();
    // printf(" time: %.6f seconds\n", (double)(end6 - start6) / CLOCKS_PER_SEC);


    free(cache);
    free(jobs);
    
}


        
void update_u2(const double *u2, const double *J, const double *tau_cd, 
               int nocc, int nvir, int naux, int nauxv, double *u2new_sum) {

    // for (int i = 0; i < nocc; i++){
    //   printf("u2[%d]: %f \n", i, u2[i]);
    //   printf("J[%d]: %f \n", i, J[i]);
    //   printf("tau_cd[%d]: %f \n", i, tau_cd[i]);
    // }

    size_t nov = nocc * nvir;
    size_t nvv = nvir * nvir;
    size_t nvxv = nvir * nauxv;

    // Allocate memory
    double *u2w = malloc(nov * nvxv * sizeof(double));
    double *vw = malloc(nauxv * nauxv * sizeof(double));
    double *u2new = malloc(nov * nauxv * sizeof(double));
    // double *u2new_sum = calloc(nov * nauxv, sizeof(double)); // Initialize to 0

    const double D0 = 0.0;
    const double D1 = 1.0;

    // Flatten tau_cd for efficient access
    double *expanded_tau_cd = malloc(naux * nov * nauxv * sizeof(double));
    for (int w = 0; w < naux; ++w) {
        for (int i = 0; i < nov; ++i) {
                for (int k = 0; k < nauxv; ++k) {
                    expanded_tau_cd[w * nov * nauxv + i * nauxv + k] = tau_cd[w * nov + i ];
                }
        }
    }

    // Main loop over w
    for (int w = 0; w < naux; ++w) {
        // Compute element-wise multiplication: u2w = u2 * expanded_tau_cd[w]
        const double *tau_w = &expanded_tau_cd[w * nov * nauxv]; // Pointer to the current tau_cd slice
        elementwise_multiply(nov * nauxv, u2, tau_w, u2w);

        // Define variables as int for dgemm_
        int nov_int = (int)nov;
        int nauxv_int = (int)nauxv;

        // Compute vw = einsum('jbB,jbA->BA', u2w, J)
        dgemm_("T", "N", &nauxv_int, &nauxv_int, &nov_int, &D1, u2w, &nov_int, J, &nov_int, &D0, vw, &nauxv_int); 
        printf("Done with the elementwise_multiply (dgemm1)\n");

        // Compute u2new = einsum('iaB,BA->iaA', u2w, vw)
        dgemm_("N", "T", &nov_int, &nauxv_int, &nauxv_int, &D1, u2w, &nov_int, vw, &nauxv_int, &D0, u2new, &nov_int);
        printf("Done with the elementwise_multiply (dgemm2)\n");
        

        // Update u2new_sum -= 2.0 * u2new
        for (size_t i = 0; i < nov * nauxv; ++i) {
            u2new_sum[i] -= 2.0 * u2new[i];
        }

    }

    // Free memory
    free(u2w);
    free(vw);
    free(u2new);
    free(expanded_tau_cd);
}