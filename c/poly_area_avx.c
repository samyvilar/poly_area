#include "poly_area.h"

#include <immintrin.h>


#define _mm256_flip_sign_ps(a) _mm256_xor_ps(a, _mm256_set1_ps(-0.0f))

#ifdef __AVX__
irreg_poly_area_func_sign(float, _avx) {
    if (__builtin_expect(is_null(cords) || cords_len == 0, 0))
        return 0;

    __m256
        values_0_3,
        values_4_7,
        values_8_11,
        values_12_15,
        values_16_19 = _mm256_load_ps((const float *)&cords[0][0]),
        accum_sum = _mm256_setzero_ps();
    float accum_sum_aux;

    #define _float_cords_dot_prod(curr, next, index)                    \
        _mm256_dp_ps(                                                   \
            curr,                                                       \
            _mm256_xor_ps(                                              \
                _mm256_shuffle_ps(curr, _mm256_permute2f128_ps(curr, next, 0b00100001), 0b00011011),\
                _mm256_setr_ps(0, -0.0f, 0, -0.0f, 0, -0.0f, 0, -0.0f)  \
            ),                                                          \
            0b11110000 | (1 << (index))                                 \
        )


    unsigned long index;
    for (index = 0; index < (cords_len - 16); index += 16) {
        values_0_3   = values_16_19;
        values_4_7   = _mm256_load_ps((const float *)&cords[index + 4]);
        values_8_11  = _mm256_load_ps((const float *)&cords[index + 8]);
        values_12_15 = _mm256_load_ps((const float *)&cords[index + 12]);
        values_16_19 = _mm256_load_ps((const float *)&cords[index + 16]);

        accum_sum = _mm256_add_ps(
            accum_sum,
            _mm256_add_ps(
                _mm256_add_ps(
                    _float_cords_dot_prod(values_0_3, values_4_7, 0),
                    _float_cords_dot_prod(values_4_7, values_8_11, 1)
                ),
                _mm256_add_ps(
                    _float_cords_dot_prod(values_8_11, values_12_15, 2),
                    _float_cords_dot_prod(values_12_15, values_16_19, 3)
                )
            )
        );
    }

    accum_sum = _mm256_hadd_ps(accum_sum, _mm256_permute2f128_ps(accum_sum, accum_sum, 1)); // a0+a1, a2+a3, a4+a5, a6+a7, a4+a5, a6+a7, a0+a1, a2+a3
    accum_sum = _mm256_hadd_ps(accum_sum, accum_sum); // a0+a1+a2+a3, a4+a5+a6+a7, ...
    accum_sum = _mm256_hadd_ps(accum_sum, accum_sum); // a0+a1+a2+a3+a4+a5+a6+a7, ...
    for (accum_sum_aux = _mm_cvtss_f32(_mm256_castps256_ps128(accum_sum)); index < (cords_len - 1); index++)
        accum_sum_aux += _calc_diff_of_adj_prods(cords, index);

    return accum_sum_aux;
//    return scalar_half(scalar_abs(accum_sum_aux));
}


irreg_poly_area_func_sign(double, _avx) {
    if (__builtin_expect(is_null(cords) || cords_len == 0, 0))
        return 0;

    __m256d
        curr,
        forw,
        coef_0,
        coef_1,
        end = _mm256_load_pd((const double *)cords),
        accum_sum = _mm256_setzero_pd();
    double accum_sum_aux;

    unsigned long index;
    for (index = 0; index < (cords_len - 4); index += 4) {
        curr = end;                                                 // x0,y0,x1,y1
        forw = _mm256_load_pd((const double *)&cords[index + 2]);   // x2,y2,x3,y3
        end = _mm256_load_pd((const double *)&cords[index + 4]);    // x4,y4,x5,y5

        coef_0 = _mm256_permute2f128_pd(curr, forw, 0b00110001); // x1, y1, x3, y3
        coef_1 = _mm256_permute2f128_pd(forw, end, 0b00100000); // x2, y2, x4, y4

        //_mm256_hsub_pd(a, b) == a0 - a1, b0 - b1, a2 - a3, b2 - b3
        accum_sum = _mm256_add_pd(
            accum_sum,
            _mm256_hsub_pd( // x0*y1 - y0*x1, x1*y2 - y1x2, x2*y3 - y2*x3, x3*y4 - y3*x4
                _mm256_mul_pd( // x0*y1, y0*x1, x2*y3, y2*x3
                    _mm256_permute2f128_pd(curr, forw, 0b00100000),  // x0, y0, x2, y2
                    _mm256_shuffle_pd(coef_0, coef_0, 0b0101)  // y1, x1, y3, x3
                ),
                _mm256_mul_pd(coef_0, _mm256_shuffle_pd(coef_1, coef_1, 0b0101)) // y2, x2, y4, x4
                // ^^^^^^^^^^^^^^^  x1*y2, y1*x2, x3*y4, y3*x4
            )
        );
    }

    accum_sum = _mm256_hadd_pd(accum_sum, _mm256_permute2f128_pd(accum_sum, accum_sum, 1)); // a0+a1, a2+a3, a2+a3, a0+a1
    accum_sum = _mm256_hadd_pd(accum_sum, accum_sum); // a0+a1+a2+a3, ...
    for (accum_sum_aux = _mm_cvtsd_f64(_mm256_castpd256_pd128(accum_sum)); index < (cords_len - 1); index++)
        accum_sum_aux += _calc_diff_of_adj_prods(cords, index);

    return accum_sum_aux;
//    return scalar_half(scalar_abs(accum_sum_aux));
}


//int main() {
//    float result_flt, temp[] __attribute__ ((aligned (32))) = {0.25, 0.25, 1.25, 0.25, 1.25, 1.25, 2.25, 1.25, 2.25, 2.25, 3.25, 2.25, 3.25, 3.25, 4.25, 3.25, 4.25, 4.25, 5.25, 4.25, 5.25, 5.25, 6.25, 5.25, 6.25, 6.25, 7.25, 6.25, 7.25, 7.25, 8.25, 7.25, 8.25, 8.25, 9.25, 8.25, 9.25, 9.25, 10.25, 9.25, 10.25, 10.25, 11.25, 10.25, 11.25, 11.25, 12.25, 11.25, 12.25, 12.25, 13.25, 12.25, 13.25, 13.25, 14.25, 13.25, 14.25, 14.25, 15.25, 14.25, 15.25, 15.25, 16.25, 15.25, 16.25, 16.25, 17.25, 16.25, 17.25, 17.25, 18.25, 17.25, 18.25, 18.25, 19.25, 18.25, 19.25, 19.25, 20.25, 19.25, 20.25, 20.25, 21.25, 20.25, 21.25, 21.25, 22.25, 21.25, 22.25, 22.25, 23.25, 22.25, 23.25, 23.25, 24.25, 23.25, 24.25, 24.25, 25.25, 24.25, 25.25, 25.25, 26.25, 25.25, 26.25, 26.25, 27.25, 26.25, 27.25, 27.25, 28.25, 27.25, 28.25, 28.25, 29.25, 28.25, 29.25, 29.25, 30.25, 29.25, 30.25, 30.25, 31.25, 30.25, 31.25, 31.25, 32.25, 31.25, 32.25, 32.25, 32.25, 33.25, 31.25, 33.25, 31.25, 32.25, 30.25, 32.25, 30.25, 31.25, 29.25, 31.25, 29.25, 30.25, 28.25, 30.25, 28.25, 29.25, 27.25, 29.25, 27.25, 28.25, 26.25, 28.25, 26.25, 27.25, 25.25, 27.25, 25.25, 26.25, 24.25, 26.25, 24.25, 25.25, 23.25, 25.25, 23.25, 24.25, 22.25, 24.25, 22.25, 23.25, 21.25, 23.25, 21.25, 22.25, 20.25, 22.25, 20.25, 21.25, 19.25, 21.25, 19.25, 20.25, 18.25, 20.25, 18.25, 19.25, 17.25, 19.25, 17.25, 18.25, 16.25, 18.25, 16.25, 17.25, 15.25, 17.25, 15.25, 16.25, 14.25, 16.25, 14.25, 15.25, 13.25, 15.25, 13.25, 14.25, 12.25, 14.25, 12.25, 13.25, 11.25, 13.25, 11.25, 12.25, 10.25, 12.25, 10.25, 11.25, 9.25, 11.25, 9.25, 10.25, 8.25, 10.25, 8.25, 9.25, 7.25, 9.25, 7.25, 8.25, 6.25, 8.25, 6.25, 7.25, 5.25, 7.25, 5.25, 6.25, 4.25, 6.25, 4.25, 5.25, 3.25, 5.25, 3.25, 4.25, 2.25, 4.25, 2.25, 3.25, 1.25, 3.25, 1.25, 2.25, 0.25, 2.25, 0.25, 1.25, 0.25, 0.25};
//    double result_dbl, temp_double[] __attribute__ ((aligned (32))) = {0.25, 0.25, 1.25, 0.25, 1.25, 1.25, 2.25, 1.25, 2.25, 2.25, 3.25, 2.25, 3.25, 3.25, 4.25, 3.25, 4.25, 4.25, 5.25, 4.25, 5.25, 5.25, 6.25, 5.25, 6.25, 6.25, 7.25, 6.25, 7.25, 7.25, 8.25, 7.25, 8.25, 8.25, 9.25, 8.25, 9.25, 9.25, 10.25, 9.25, 10.25, 10.25, 11.25, 10.25, 11.25, 11.25, 12.25, 11.25, 12.25, 12.25, 13.25, 12.25, 13.25, 13.25, 14.25, 13.25, 14.25, 14.25, 15.25, 14.25, 15.25, 15.25, 16.25, 15.25, 16.25, 16.25, 17.25, 16.25, 17.25, 17.25, 18.25, 17.25, 18.25, 18.25, 19.25, 18.25, 19.25, 19.25, 20.25, 19.25, 20.25, 20.25, 21.25, 20.25, 21.25, 21.25, 22.25, 21.25, 22.25, 22.25, 23.25, 22.25, 23.25, 23.25, 24.25, 23.25, 24.25, 24.25, 25.25, 24.25, 25.25, 25.25, 26.25, 25.25, 26.25, 26.25, 27.25, 26.25, 27.25, 27.25, 28.25, 27.25, 28.25, 28.25, 29.25, 28.25, 29.25, 29.25, 30.25, 29.25, 30.25, 30.25, 31.25, 30.25, 31.25, 31.25, 32.25, 31.25, 32.25, 32.25, 32.25, 33.25, 31.25, 33.25, 31.25, 32.25, 30.25, 32.25, 30.25, 31.25, 29.25, 31.25, 29.25, 30.25, 28.25, 30.25, 28.25, 29.25, 27.25, 29.25, 27.25, 28.25, 26.25, 28.25, 26.25, 27.25, 25.25, 27.25, 25.25, 26.25, 24.25, 26.25, 24.25, 25.25, 23.25, 25.25, 23.25, 24.25, 22.25, 24.25, 22.25, 23.25, 21.25, 23.25, 21.25, 22.25, 20.25, 22.25, 20.25, 21.25, 19.25, 21.25, 19.25, 20.25, 18.25, 20.25, 18.25, 19.25, 17.25, 19.25, 17.25, 18.25, 16.25, 18.25, 16.25, 17.25, 15.25, 17.25, 15.25, 16.25, 14.25, 16.25, 14.25, 15.25, 13.25, 15.25, 13.25, 14.25, 12.25, 14.25, 12.25, 13.25, 11.25, 13.25, 11.25, 12.25, 10.25, 12.25, 10.25, 11.25, 9.25, 11.25, 9.25, 10.25, 8.25, 10.25, 8.25, 9.25, 7.25, 9.25, 7.25, 8.25, 6.25, 8.25, 6.25, 7.25, 5.25, 7.25, 5.25, 6.25, 4.25, 6.25, 4.25, 5.25, 3.25, 5.25, 3.25, 4.25, 2.25, 4.25, 2.25, 3.25, 1.25, 3.25, 1.25, 2.25, 0.25, 2.25, 0.25, 1.25, 0.25, 0.25};
//    if ((result_flt = irreg_poly_area_avx_float((float (*)[2])&temp, sizeof(temp)/8)) != 64.0f)
//        printf("test failed got %f!\n", result_flt);

//    if ((result_dbl = irreg_poly_area_avx_double((double (*)[2])&temp_double, sizeof(temp_double)/16)) != 64.0)
//        printf("test failed got %f!\n", result_dbl);

//    return 0;
//}
#endif
