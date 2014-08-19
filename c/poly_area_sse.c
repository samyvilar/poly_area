
#include "poly_area.h"


irreg_poly_area_func_sign(float, _sse) {
    if (__builtin_expect(is_null(cords) || cords_len == 0, 0))
        return 0;

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
}


irreg_poly_area_func_sign(double, _sse) {
    if (__builtin_expect(is_null(cords) || cords_len == 0, 0))
        return 0;

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
}
