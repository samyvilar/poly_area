
#include <limits.h>
#include <stdlib.h>
#include <immintrin.h>

#define is_null(v) (v == NULL)
#define bit_size(expr) (sizeof(expr) * CHAR_BIT)
#define re_interp(expr, from_type, to_type) (((union {from_type _; to_type interp_expr;}){(expr)}).interp_expr)

#define word_t long
#define uword_t unsigned word_t

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


#define _mm_abs_ps(v) _mm_andnot_ps(_mm_set1_ps(-0.0f), v)

#define calc_area_from_sum_of_prods(x) (x) // scalar_half(scalar_abs(accum_sum_aux))

#define irreg_poly_area_impl_name(base_type, intrs)  irreg_poly_area ## intrs ## _ ## base_type
#define irreg_poly_area_func_sign(base_type, intrs)         \
    base_type irreg_poly_area_impl_name(base_type, intrs)(  \
        base_type cords[][2],                               \
        unsigned long cords_len                             \
    )

irreg_poly_area_func_sign(float,);
irreg_poly_area_func_sign(double,);

irreg_poly_area_func_sign(float, _sse);
irreg_poly_area_func_sign(double, _sse);

#ifdef __AVX__

    irreg_poly_area_func_sign(float, _avx);
    irreg_poly_area_func_sign(double, _avx);


#endif



