/* Benchmark-only LP64 -> NumPy wheel ILP64 adapter. No arithmetic or fallback.
 * The enum arguments remain C int; only matrix dimensions become int64_t.
 * This binds private scipy_* symbols and is deliberately not a default backend.
 */
#include <stdint.h>

extern void scipy_cblas_dgemm64_(int, int, int, int64_t, int64_t, int64_t,
    double, const double *, int64_t, const double *, int64_t, double, double *, int64_t);
extern void scipy_cblas_dsyrk64_(int, int, int, int64_t, int64_t,
    double, const double *, int64_t, double, double *, int64_t);

void cblas_dgemm(int order, int ta, int tb, int m, int n, int k,
    double alpha, const double *a, int lda, const double *b, int ldb,
    double beta, double *c, int ldc) {
    scipy_cblas_dgemm64_(order, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void cblas_dsyrk(int order, int uplo, int trans, int n, int k,
    double alpha, const double *a, int lda, double beta, double *c, int ldc) {
    scipy_cblas_dsyrk64_(order, uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}
