
#include <stdlib.h>
#include <stdio.h>

#include <limits.h>

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
            sum_of_diffs += (cord_x(cords[index]) * cord_y(cords[index + 1]))   \
                          - (cord_y(cords[index]) * cord_x(cords[index + 1]));  \
        return scalar_half(scalar_abs(sum_of_diffs));                           \
    }

area_of_irregular_polygon_from_cords_tmpl(double, double)

area_of_irregular_polygon_from_cords_tmpl(float, float)
