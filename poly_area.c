#include <limits.h>
#include <stdlib.h>
#include <stdio.h>
#include <immintrin.h>

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

#define scalar_abs(expr) ({                                                                         \
    typedef typeof(expr) __expr_t_abs_;                                                             \
    select_expr(type_eq(__expr_t_abs_, double), double_abs(expr),                                 \
    select_expr(type_eq(__expr_t_abs_, float), float_abs(expr),                                   \
        error_applying_unsupported_type())); })
//    select_expr(type_eq(__expr_t_abs_, signed char) || is_expr_t(__expr_t_abs_, char), (unsigned char)sint_abs(expr),\
//    select_expr(type_eq(__expr_t_abs_, short), (unsigned short)sint_abs(expr),                    \
//    select_expr(type_eq(__expr_t_abs_, int), (unsigned int)sint_abs(expr),                     \
//    select_expr(type_eq(__expr_t_abs_, signed long int), (unsigned int)sint_abs(expr),         \
//    select_expr(type_eq(__expr_t_abs_, long long int), (unsigned long long)sint_abs(expr),     \
//        expr)))))));                                                                                \
//})


// we can a divide a float by 2, assuming it is sufficiently large, by subtracting 1 from its exponent ....
#define DOUBLE_INDEX_OF_EXPONENT 52
#define FLOAT_INDEX_OF_EXPONENT 23
#define double_half(expr) re_interp((re_interp(expr, double, long long) - (1LLU << DOUBLE_INDEX_OF_EXPONENT)), long long, double)
#define float_half(expr)  re_interp((re_interp(expr, float, int) - (1 << FLOAT_INDEX_OF_EXPONENT)), int, float)
//#define integral_half(expr) ((expr) >> 1)

#define scalar_half(expr)                                               \
    select_expr(type_eq(typeof(expr), double), double_half(expr),       \
    select_expr(type_eq(typeof(expr), float), float_half(expr),         \
        error_applying_unsupported_type()))
//        integral_half(expr)))


#define cord_x(v) (v[0])
#define cord_y(v) (v[1])

#define _calc_diff_of_adj_prods(cords, index) \
    ((cord_x(cords[index]) * cord_y(cords[index + 1])) - (cord_y(cords[index]) * cord_x(cords[index + 1])))


// calculate area of an irregular polygon using its flatten array of its coordinates ...
#define area_of_irregular_polygon_from_cords_tmpl(member_type, prefix_name)     \
    member_type area_of_irregular_polygon_from_cords_ ## prefix_name(           \
        member_type cords[][2],                                                 \
        unsigned long cords_len                                                 \
    ) {                                                                         \
        if (__builtin_expect(is_null(cords) || cords_len == 0, 0))              \
            return 0;                                                           \
        member_type sum_of_diffs = 0;                                           \
        unsigned long index;                                                    \
        for (index = 0; index < (cords_len - 1); index++)                       \
            sum_of_diffs += _calc_diff_of_adj_prods(cords, index);              \
        return scalar_half(scalar_abs(sum_of_diffs));                           \
    }

area_of_irregular_polygon_from_cords_tmpl(double, double)

area_of_irregular_polygon_from_cords_tmpl(float, float)



#define _mm_abs_ps(v) _mm_andnot_ps(_mm_set1_ps(-0.0f), v)
#ifndef __SSE3__
//    dst[31:0] := a[63:32] + a[31:0]
//    dst[63:32] := a[127:96] + a[95:64]
//    dst[95:64] := b[63:32] + b[31:0]
//    dst[127:96] := b[127:96] + b[95:64]
    #define _mm_hadd_ps(a, b)                               \
        _mm_add_ps(                                         \
            _mm_shuffle_ps(a, b, _MM_SHUFFLE(3, 1, 3, 1)),  \
            _mm_shuffle_ps(a, b, _MM_SHUFFLE(2, 0, 2, 0))   \
        )
#endif

// x_0, y_0, x_1, y_1
// x_1, y_1, x_2, y_2
// y_1, x_1, y_2, x_2
//1) load x_0, y_0, x_1, y_1
//2) load x_1, y_1, x_2, y_2
//3) shuf y_1, x_1, y_2, x_2
//4) mul x_0*y_1, y_0*x_1, x_1*y_2, y_1*x_2

// x_0*y_1 - y_0*x_1, x_1*y_2 - y_1*x_2,

// x_2*y_3 - y_2*x_3, x_3*y_4 - y_3*x_4

float area_of_irregular_polygon_from_cords_sse_float(float cords[][2], unsigned long cords_len) {
    if (__builtin_expect(is_null(cords) || cords_len == 0, 0))
        return 0;

    __m128 accum_sum = _mm_setzero_ps();
    float accum_sum_aux = 0;
    unsigned long index;

    for (index = 0; (index + 4) < (cords_len - 1); index += 4) {
        __m128 low = _mm_load_ps((const float *)cords + index), // x0, y0, x1, y1
               high = _mm_load_ps((const float *)cords + index + 4), // x2, y2, x3, y3
               end = _mm_load_ps((const float *)cords + index + 8); // x4 y4

        accum_sum = _mm_add_ps(
            accum_sum,
            _mm_hsub_ps(
                _mm_mul_ps(low, _mm_shuffle_ps(low, high, _MM_SHUFFLE(0, 1, 2, 3))),
                _mm_mul_ps(high, _mm_shuffle_ps(high, end, _MM_SHUFFLE(0, 1, 2, 3)))
            )
        );
    }

    accum_sum = _mm_hadd_ps(accum_sum, accum_sum);
    accum_sum = _mm_hadd_ps(accum_sum, accum_sum);
    accum_sum_aux += _mm_cvtss_f32(accum_sum);

    for (; index < (cords_len - 1); index++)
        accum_sum_aux += _calc_diff_of_adj_prods(cords, index);

    return scalar_half(scalar_abs(accum_sum_aux));
}

#ifdef __AVX__
float area_of_irregular_polygon_from_cords_avx_float(float cords[][2], unsigned long cords_len) {
    if (__builtin_expect(is_null(cords) || cords_len == 0, 0))
        return 0;

    __m256 accum_sum = _mm256_setzero_ps();
    float accum_sum_aux = 0;
    unsigned long index;

    for (index = 0; (index + 8) < (cords_len - 1); index += 8) {
        typeof(accum_sum)
                low = _mm256_load_ps((const float *)cords + index), // x0, y0, x1, y1, x2, y2, x3, y3
                high = _mm256_load_ps((const float *)cords + index + 8), // x4, y4 ... x7, y7 ...
                end = _mm256_load_ps((const float *)cords + index + 16); // x8, y8

        accum_sum = _mm256_add_ps(
            accum_sum,
            _mm256_hsub_ps(
                _mm256_mul_ps(low, _mm256_shuffle_ps(low, high, _MM_SHUFFLE(0, 1, 2, 3))),
                _mm256_mul_ps(high, _mm256_shuffle_ps(high, end, _MM_SHUFFLE(0, 1, 2, 3)))
            )
        );
    }

    accum_sum = _mm256_hadd_ps(accum_sum, accum_sum); // add 8
    accum_sum = _mm256_hadd_ps(accum_sum, accum_sum); // add 4
    accum_sum_aux += _mm_cvtss_f32(_mm256_extractf128_ps(accum_sum, 0));

    for (; index < (cords_len - 1); index++)
        accum_sum_aux += _calc_diff_of_adj_prods(cords, index);

    return scalar_half(scalar_abs(accum_sum_aux));
}


int main() {

    float temp[] = {0.25, 0.25, 1.25, 0.25, 1.25, 1.25, 2.25, 1.25, 2.25, 2.25, 3.25, 2.25, 3.25, 3.25, 4.25, 3.25, 4.25, 4.25, 5.25, 4.25, 5.25, 5.25, 6.25, 5.25, 6.25, 6.25, 7.25, 6.25, 7.25, 7.25, 8.25, 7.25, 8.25, 8.25, 9.25, 8.25, 9.25, 9.25, 10.25, 9.25, 10.25, 10.25, 11.25, 10.25, 11.25, 11.25, 12.25, 11.25, 12.25, 12.25, 13.25, 12.25, 13.25, 13.25, 14.25, 13.25, 14.25, 14.25, 15.25, 14.25, 15.25, 15.25, 16.25, 15.25, 16.25, 16.25, 17.25, 16.25, 17.25, 17.25, 18.25, 17.25, 18.25, 18.25, 19.25, 18.25, 19.25, 19.25, 20.25, 19.25, 20.25, 20.25, 21.25, 20.25, 21.25, 21.25, 22.25, 21.25, 22.25, 22.25, 23.25, 22.25, 23.25, 23.25, 24.25, 23.25, 24.25, 24.25, 25.25, 24.25, 25.25, 25.25, 26.25, 25.25, 26.25, 26.25, 27.25, 26.25, 27.25, 27.25, 28.25, 27.25, 28.25, 28.25, 29.25, 28.25, 29.25, 29.25, 30.25, 29.25, 30.25, 30.25, 31.25, 30.25, 31.25, 31.25, 32.25, 31.25, 32.25, 32.25, 32.25, 33.25, 31.25, 33.25, 31.25, 32.25, 30.25, 32.25, 30.25, 31.25, 29.25, 31.25, 29.25, 30.25, 28.25, 30.25, 28.25, 29.25, 27.25, 29.25, 27.25, 28.25, 26.25, 28.25, 26.25, 27.25, 25.25, 27.25, 25.25, 26.25, 24.25, 26.25, 24.25, 25.25, 23.25, 25.25, 23.25, 24.25, 22.25, 24.25, 22.25, 23.25, 21.25, 23.25, 21.25, 22.25, 20.25, 22.25, 20.25, 21.25, 19.25, 21.25, 19.25, 20.25, 18.25, 20.25, 18.25, 19.25, 17.25, 19.25, 17.25, 18.25, 16.25, 18.25, 16.25, 17.25, 15.25, 17.25, 15.25, 16.25, 14.25, 16.25, 14.25, 15.25, 13.25, 15.25, 13.25, 14.25, 12.25, 14.25, 12.25, 13.25, 11.25, 13.25, 11.25, 12.25, 10.25, 12.25, 10.25, 11.25, 9.25, 11.25, 9.25, 10.25, 8.25, 10.25, 8.25, 9.25, 7.25, 9.25, 7.25, 8.25, 6.25, 8.25, 6.25, 7.25, 5.25, 7.25, 5.25, 6.25, 4.25, 6.25, 4.25, 5.25, 3.25, 5.25, 3.25, 4.25, 2.25, 4.25, 2.25, 3.25, 1.25, 3.25, 1.25, 2.25, 0.25, 2.25, 0.25, 1.25, 0.25, 0.25};
    if (area_of_irregular_polygon_from_cords_avx_float((float *)temp, sizeof(temp)/8) != 64.0f)
        printf("test failed got %f!\n", area_of_irregular_polygon_from_cords_avx_float((float *)temp, sizeof(temp)/8));

    return 0;
}
#else
float area_of_irregular_polygon_from_cords_avx_float(float cords[][2], unsigned long cords_len) {
    printf("NO AVX SUPPORT!\n");
    exit(-1);
}
#endif







