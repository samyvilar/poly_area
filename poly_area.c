#include <limits.h>
#include <stdlib.h>
#include <stdio.h>
#include <immintrin.h>
#include <string.h>

#define is_null(v) (v == NULL)
#define bit_size(expr) (sizeof(expr) * CHAR_BIT)
#define re_interp(expr, from_type, to_type) (((union {from_type _; to_type interp_expr;}){(expr)}).interp_expr)


// twos complement to get abs(signed integral expr) a < 0 ==> ~a + 1, a >= 0 ==> a, (a ^ (a >> 32)) - (a >> 32)
//#define twos_complement(sint_expr) (sint_expr ^ (bit_size(sint_expr) - 1)) - ((sint_expr) >> (bit_size(sint_expr) - 1))

// to get the absolute value of a float all we need to do is set the significant bit to 0
#define flt_abs(x, flt_type, intgl_type) re_interp(\
    re_interp(x, flt_type, intgl_type) & ~(((intgl_type)1 << (bit_size(flt_type) - 1))),\
    intgl_type,     \
    flt_type        \
)
#define double_abs(x)   flt_abs(x, double, unsigned long)
#define float_abs(x)    flt_abs(x, float, unsigned)
//#define sint_abs twos_complement
//#define uint_abs(x) (x)

#define type_eq __builtin_types_compatible_p
#define select_expr __builtin_choose_expr

extern void error_applying_unsupported_type();

#define scalar_abs(expr) ({                                         \
    typedef typeof(expr) __expr_t_abs_;                             \
    select_expr(type_eq(__expr_t_abs_, double), double_abs(expr),   \
    select_expr(type_eq(__expr_t_abs_, float), float_abs(expr),     \
        error_applying_unsupported_type())); })

// we can a divide a float by 2, assuming it is sufficiently large, by subtracting 1 from its exponent ....
#define DOUBLE_INDEX_OF_EXPONENT 52
#define FLOAT_INDEX_OF_EXPONENT 23
#define double_half(expr) re_interp((re_interp(expr, double, long long) - (1LLU << DOUBLE_INDEX_OF_EXPONENT)), long long, double)
#define float_half(expr)  re_interp((re_interp(expr, float, int) - (1 << FLOAT_INDEX_OF_EXPONENT)), int, float)
//#define integral_half(expr) ((expr) >> 1)

#define scalar_half(expr)                                           \
    select_expr(type_eq(typeof(expr), double), double_half(expr),   \
    select_expr(type_eq(typeof(expr), float), float_half(expr),     \
        error_applying_unsupported_type()))


#define cord_x(v) ((v)[0])
#define cord_y(v) ((v)[1])

#define _calc_diff_of_adj_prods(cords, index) \
    ((cord_x(cords[index]) * cord_y(cords[(index) + 1])) - (cord_y(cords[index]) * cord_x(cords[(index) + 1])))


// calculate area of an irregular polygon using its flatten array of its coordinates ...
#define area_of_irregular_polygon_from_cords_tmpl(member_type, prefix_name)     \
    member_type irreg_poly_area_ ## prefix_name(                                \
        member_type cords[][2],                                                 \
        unsigned long cords_len                                                 \
    ) {                                                                         \
        /*if (__builtin_expect(is_null(cords) || cords_len == 0, 0))              \
            return 0;*/                                                           \
        member_type sum_of_diffs = 0;                                           \
        unsigned long index;                                                    \
        for (index = 0; index < (cords_len - 1); index++)                       \
            sum_of_diffs += _calc_diff_of_adj_prods(cords, index);              \
        return sum_of_diffs;/*return scalar_half(scalar_abs(sum_of_diffs));*/                           \
    }

area_of_irregular_polygon_from_cords_tmpl(double, double)

area_of_irregular_polygon_from_cords_tmpl(float, float)

#define _mm_abs_ps(v) _mm_andnot_ps(_mm_set1_ps(-0.0f), v)


float irreg_poly_area_sse_float(float cords[][2], unsigned long cords_len) {
//    if (__builtin_expect(is_null(cords) || cords_len == 0, 0))
//        return 0;

    __m128
        curr,
        next,
        end = _mm_load_ps((const float *)&cords[0]),
        accum_sum = _mm_setzero_ps();
    float accum_sum_aux;

    unsigned long index;
    for (index = 0; index < (cords_len - 4); index += 4) { // @@ this will fail if cords_len < 4!
        curr = end;
        next = _mm_load_ps((const float *)&cords[index + 2]);
        end = _mm_load_ps((const float *)&cords[index + 4]);

        accum_sum = _mm_add_ps( // accumulate differences ...
            accum_sum,
            _mm_hsub_ps( // x0*y1 - y0*x1, x1*y2 - y1*x2, x2*y3 - y2*x3, x3*y4 - y3*x4
                _mm_mul_ps(curr, _mm_shuffle_ps(curr, next, _MM_SHUFFLE(0, 1, 2, 3))), // x0*y1, y0*x1, x1*y2, y1*x2
                _mm_mul_ps(next, _mm_shuffle_ps(next, end, _MM_SHUFFLE(0, 1, 2, 3))) // x2*y3, y2*x3, x3*y4, y3*x4
            )
        );
    }

    accum_sum = _mm_hadd_ps(accum_sum, accum_sum);
    accum_sum = _mm_hadd_ps(accum_sum, accum_sum);
    for (accum_sum_aux = _mm_cvtss_f32(accum_sum); index < (cords_len - 1); index++)
        accum_sum_aux += _calc_diff_of_adj_prods(cords, index);

    return accum_sum_aux;
//    return scalar_half(scalar_abs(accum_sum_aux));
}

double irreg_poly_area_sse_double(double cords[][2], unsigned long cords_len) {
//    if (__builtin_expect(is_null(cords) || cords_len == 0, 0))
//        return 0;

    __m128d curr, next, end = _mm_load_pd((const double *)&cords[0]), accum_sum = _mm_setzero_pd();
    double accum_sum_aux;

    unsigned long index;
    for (index = 0; index < (cords_len - 2); index += 2) {
        curr = end; // x0, y0
        next = _mm_load_pd((const double *)&cords[index + 1]); // x1, y1
        end = _mm_load_pd((const double *)&cords[index + 2]); // x2, y2

        accum_sum = _mm_add_pd(
            accum_sum,
            _mm_hsub_pd( //y0*x1 - x0*y1, y1*x2 - x1*y2
                _mm_mul_pd(curr, _mm_shuffle_pd(next, next, _MM_SHUFFLE2(0, 1))), // x0*y1, y0*x1
                _mm_mul_pd(next, _mm_shuffle_pd(end, end, _MM_SHUFFLE2(0, 1))) // x1*y2, y1*x2
            )
        );
    }

    for (accum_sum_aux = _mm_cvtsd_f64(_mm_hadd_pd(accum_sum, accum_sum)); index < (cords_len - 1); index++)
        accum_sum_aux += _calc_diff_of_adj_prods(cords, index);

    return accum_sum_aux;
//    return scalar_half(scalar_abs(accum_sum_aux));
}

#define _mm256_flip_sign_ps(a) _mm256_xor_ps(a, _mm256_set1_ps(-0.0f))
#if 1// ndef __AVX__
float irreg_poly_area_avx_float(float cords[][2], unsigned long cords_len) {
//    if (__builtin_expect(is_null(cords) || cords_len == 0, 0))
//        return 0;

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


double irreg_poly_area_avx_double(double cords[][2], unsigned long cords_len) {
//    if (__builtin_expect(is_null(cords) || cords_len == 0, 0))
//        return 0;

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






