
#include <stdio.h>

#include "poly_area.h"



// calculate area of an irregular polygon using its flatten array of its coordinates ...
#define area_of_irregular_polygon_from_cords_tmpl(member_type)                  \
    irreg_poly_area_func_sign(member_type,) {                                   \
        if (__builtin_expect(is_null(cords) || cords_len == 0, 0))              \
            return 0;                                                           \
        member_type sum_of_diffs = 0;                                           \
        unsigned long index;                                                    \
        for (index = 0; index < (cords_len - 1); index++)                       \
            sum_of_diffs += _calc_diff_of_adj_prods(cords, index);              \
        return calc_area_from_sum_of_prods(sum_of_diffs);                       \
    }


area_of_irregular_polygon_from_cords_tmpl(double)

area_of_irregular_polygon_from_cords_tmpl(float)
