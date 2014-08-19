
#include <unistd.h>
#include <stdio.h>

#ifdef __MACH__
    #include <pthread.h>
    #include <mach/mach.h>
    #include <mach/thread_policy.h>
#else
    #define __USE_GNU
    #include <pthread.h>
    #include <sys/syscall.h>
    #include <sched.h>
#endif

#include "poly_area.h"

#define cpu_count() sysconf(_SC_NPROCESSORS_ONLN)

#define print_msg_and_exit(msg...) (printf(msg), exit(-1))

#define thrd_cords_t_name(base_type) thrd_cords_ ## base_type ## _t

#define thrd_cords_t_def_tmpl(base_type)                    \
    typedef struct thrd_cords_t_name(base_type) {           \
        void *cords;                                        \
        base_type                                           \
            (*impl)(base_type [][2], unsigned long),        \
            sum;                                            \
        unsigned long len;                                  \
        int thrd_affinity_id;                               \
    } thrd_cords_t_name(base_type);

extern void error_thrd_cords_t_not_supported();
#define thrd_cords_t(expr_or_t) typeof(                         \
    __builtin_choose_expr(                                      \
        __builtin_types_compatible_p(typeof(expr_or_t), float), \
        (thrd_cords_t_name(float)){},                           \
    __builtin_choose_expr(                                      \
        __builtin_types_compatible_p(typeof(expr_or_t), double),\
        (thrd_cords_t_name(double)){},                          \
    error_thrd_cords_t_not_supported()))                        \
)


#define thrd_cord_calc_area_name(base_type) _calc_irreg_poly_area_ ## base_type

#define ERROR_STR_FAILED_TO_SET_AFFINITY "failed to set thread affinity ..."

#define mac_set_my_affinity(aff_id) ({                                  \
    thread_extended_policy_data_t thread_policy = {.timeshare = FALSE}; \
    thread_affinity_policy_data_t policy = {.affinity_tag = aff_id};    \
    kern_return_t ret;                                                  \
    if ((ret = thread_policy_set(mach_thread_self(), THREAD_EXTENDED_POLICY, (thread_policy_t)&thread_policy, THREAD_EXTENDED_POLICY_COUNT)) != KERN_SUCCESS)\
        mach_error(ERROR_STR_FAILED_TO_SET_AFFINITY, ret);              \
    if ((ret = thread_policy_set(mach_thread_self(), THREAD_AFFINITY_POLICY, (thread_policy_t)&policy, THREAD_AFFINITY_POLICY_COUNT)) != KERN_SUCCESS)\
        mach_error(ERROR_STR_FAILED_TO_SET_AFFINITY, ret);              \
}) // ^^^^^^ Set the thread affinity on OS X

#define linux_set_my_affinity(aff_id) ({                        \
    cpu_set_t set;                                              \
    CPU_ZERO(&set);                                             \
    CPU_SET(aff_id, &set);                                      \
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &set))   \
        perror(ERROR_STR_FAILED_TO_SET_AFFINITY);               \
}) // ^^^^^ Sets the current threads affinity on Linux

#ifdef __MACH__
    #define set_my_affinity mac_set_my_affinity
#elif defined(__linux)
    #define set_my_affinity linux_set_my_affinity
#else
    #define set_my_affinity(...) ({extern void set_my_affinity_not_supported(); set_my_affinity_not_supported();})
#endif



// Setting thread affinity did not yield better restults, it seems OS X is better at scheduling them
// for this example anyway ...
#define thrd_cords_calc_area_wrapper_tmpl(base_type)                    \
    void *thrd_cord_calc_area_name(base_type) (void *sgmnt) {           \
        typedef thrd_cords_t(base_type) cords_t;                        \
        /*set_my_affinity(((cords_t *)sgmnt)->thrd_affinity_id);*/          \
        ((cords_t *)sgmnt)->sum = ((cords_t *)sgmnt)->impl(((cords_t *)sgmnt)->cords, ((cords_t *)sgmnt)->len);\
        return NULL;                                                    \
    }

thrd_cords_t_def_tmpl(float)

thrd_cords_t_def_tmpl(double)


thrd_cords_calc_area_wrapper_tmpl(float)


thrd_cords_calc_area_wrapper_tmpl(double)



#define irreg_poly_area_multi_cpu_tmpl(base_type, intrsic)                  \
base_type irreg_poly_area ## intrsic ## _ ## base_type ## _thrd(            \
    base_type cords[][2],                                                   \
    unsigned long cords_len                                                 \
){                                                                          \
    if (__builtin_expect(is_null(cords) || cords_len == 0, 0))              \
        return 0;                                                           \
    typeof(cpu_count()) segmnt_cnt = cpu_count();                           \
    thrd_cords_t(base_type) segments[segmnt_cnt];                           \
    base_type sum = 0.0f;                                                   \
    pthread_t threads[segmnt_cnt];                                          \
    /*int thrd_aff_ids[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};*/\
    unsigned long index, remainder, min_segment_size = cords_len/segmnt_cnt;\
    for (index = 0; index < segmnt_cnt; index++) {                          \
        segments[index].len = min_segment_size;                             \
        segments[index].cords = &cords[index * min_segment_size];           \
        segments[index].sum = 0;                                            \
        segments[index].impl = &irreg_poly_area_impl_name(base_type, intrsic);\
        /*segments[index].thrd_affinity_id = thrd_aff_ids[index];*/             \
    }                                                                       \
    /* ^^^^^ init thread args with segmented input */                       \
    segments[index - 1].len += (cords_len % segmnt_cnt);                    \
    /* ^^^^^ add any remainders to the very last entry ...*/                \
    for (index = 1; index < segmnt_cnt; index++) {                          \
        remainder = (uword_t)segments[index].cords % 32LLU;                 \
        segments[index].cords -= (remainder * sizeof(cords[0]));            \
        segments[index].len += remainder;                                   \
        segments[index - 1].len -= remainder;                               \
    }                                                                       \
    /* ^^^^^ aligng each segment TODO: check results. */                    \
    for (index = 0; index < (segmnt_cnt - 1); index++)                      \
        segments[index].len++;                                              \
    /* ^^^^^^ increment segments on all but very last entry */              \
    for (index = 0; index < segmnt_cnt; index++)                            \
        if (pthread_create(&threads[index], NULL, thrd_cord_calc_area_name(base_type), &segments[index]))\
            print_msg_and_exit("failed to create thread\n");                \
    /* ^^^^^ create and start threads ... */                                \
    for (index = 0; index < segmnt_cnt; index++)                            \
        if (pthread_join(threads[index], NULL))                             \
            print_msg_and_exit("failed to thread_join()\n");                \
        else                                                                \
            sum += segments[index].sum;                                     \
    /* ^^^^^ wait for thread(s) to terminate, add up results ... */         \
    return sum;                                                             \
}

irreg_poly_area_multi_cpu_tmpl(float,)

irreg_poly_area_multi_cpu_tmpl(double,)


irreg_poly_area_multi_cpu_tmpl(float, _sse)

irreg_poly_area_multi_cpu_tmpl(double, _sse)


#ifdef __AVX__
    irreg_poly_area_multi_cpu_tmpl(float, _avx)

    irreg_poly_area_multi_cpu_tmpl(double, _avx)
#endif

//int main() {
//    float result_flt, temp[] __attribute__ ((aligned (32))) = {0.25, 0.25, 1.25, 0.25, 1.25, 1.25, 2.25, 1.25, 2.25, 2.25, 3.25, 2.25, 3.25, 3.25, 4.25, 3.25, 4.25, 4.25, 5.25, 4.25, 5.25, 5.25, 6.25, 5.25, 6.25, 6.25, 7.25, 6.25, 7.25, 7.25, 8.25, 7.25, 8.25, 8.25, 9.25, 8.25, 9.25, 9.25, 10.25, 9.25, 10.25, 10.25, 11.25, 10.25, 11.25, 11.25, 12.25, 11.25, 12.25, 12.25, 13.25, 12.25, 13.25, 13.25, 14.25, 13.25, 14.25, 14.25, 15.25, 14.25, 15.25, 15.25, 16.25, 15.25, 16.25, 16.25, 17.25, 16.25, 17.25, 17.25, 18.25, 17.25, 18.25, 18.25, 19.25, 18.25, 19.25, 19.25, 20.25, 19.25, 20.25, 20.25, 21.25, 20.25, 21.25, 21.25, 22.25, 21.25, 22.25, 22.25, 23.25, 22.25, 23.25, 23.25, 24.25, 23.25, 24.25, 24.25, 25.25, 24.25, 25.25, 25.25, 26.25, 25.25, 26.25, 26.25, 27.25, 26.25, 27.25, 27.25, 28.25, 27.25, 28.25, 28.25, 29.25, 28.25, 29.25, 29.25, 30.25, 29.25, 30.25, 30.25, 31.25, 30.25, 31.25, 31.25, 32.25, 31.25, 32.25, 32.25, 32.25, 33.25, 31.25, 33.25, 31.25, 32.25, 30.25, 32.25, 30.25, 31.25, 29.25, 31.25, 29.25, 30.25, 28.25, 30.25, 28.25, 29.25, 27.25, 29.25, 27.25, 28.25, 26.25, 28.25, 26.25, 27.25, 25.25, 27.25, 25.25, 26.25, 24.25, 26.25, 24.25, 25.25, 23.25, 25.25, 23.25, 24.25, 22.25, 24.25, 22.25, 23.25, 21.25, 23.25, 21.25, 22.25, 20.25, 22.25, 20.25, 21.25, 19.25, 21.25, 19.25, 20.25, 18.25, 20.25, 18.25, 19.25, 17.25, 19.25, 17.25, 18.25, 16.25, 18.25, 16.25, 17.25, 15.25, 17.25, 15.25, 16.25, 14.25, 16.25, 14.25, 15.25, 13.25, 15.25, 13.25, 14.25, 12.25, 14.25, 12.25, 13.25, 11.25, 13.25, 11.25, 12.25, 10.25, 12.25, 10.25, 11.25, 9.25, 11.25, 9.25, 10.25, 8.25, 10.25, 8.25, 9.25, 7.25, 9.25, 7.25, 8.25, 6.25, 8.25, 6.25, 7.25, 5.25, 7.25, 5.25, 6.25, 4.25, 6.25, 4.25, 5.25, 3.25, 5.25, 3.25, 4.25, 2.25, 4.25, 2.25, 3.25, 1.25, 3.25, 1.25, 2.25, 0.25, 2.25, 0.25, 1.25, 0.25, 0.25};
//    double result_dbl, temp_double[] __attribute__ ((aligned (32))) = {0.25, 0.25, 1.25, 0.25, 1.25, 1.25, 2.25, 1.25, 2.25, 2.25, 3.25, 2.25, 3.25, 3.25, 4.25, 3.25, 4.25, 4.25, 5.25, 4.25, 5.25, 5.25, 6.25, 5.25, 6.25, 6.25, 7.25, 6.25, 7.25, 7.25, 8.25, 7.25, 8.25, 8.25, 9.25, 8.25, 9.25, 9.25, 10.25, 9.25, 10.25, 10.25, 11.25, 10.25, 11.25, 11.25, 12.25, 11.25, 12.25, 12.25, 13.25, 12.25, 13.25, 13.25, 14.25, 13.25, 14.25, 14.25, 15.25, 14.25, 15.25, 15.25, 16.25, 15.25, 16.25, 16.25, 17.25, 16.25, 17.25, 17.25, 18.25, 17.25, 18.25, 18.25, 19.25, 18.25, 19.25, 19.25, 20.25, 19.25, 20.25, 20.25, 21.25, 20.25, 21.25, 21.25, 22.25, 21.25, 22.25, 22.25, 23.25, 22.25, 23.25, 23.25, 24.25, 23.25, 24.25, 24.25, 25.25, 24.25, 25.25, 25.25, 26.25, 25.25, 26.25, 26.25, 27.25, 26.25, 27.25, 27.25, 28.25, 27.25, 28.25, 28.25, 29.25, 28.25, 29.25, 29.25, 30.25, 29.25, 30.25, 30.25, 31.25, 30.25, 31.25, 31.25, 32.25, 31.25, 32.25, 32.25, 32.25, 33.25, 31.25, 33.25, 31.25, 32.25, 30.25, 32.25, 30.25, 31.25, 29.25, 31.25, 29.25, 30.25, 28.25, 30.25, 28.25, 29.25, 27.25, 29.25, 27.25, 28.25, 26.25, 28.25, 26.25, 27.25, 25.25, 27.25, 25.25, 26.25, 24.25, 26.25, 24.25, 25.25, 23.25, 25.25, 23.25, 24.25, 22.25, 24.25, 22.25, 23.25, 21.25, 23.25, 21.25, 22.25, 20.25, 22.25, 20.25, 21.25, 19.25, 21.25, 19.25, 20.25, 18.25, 20.25, 18.25, 19.25, 17.25, 19.25, 17.25, 18.25, 16.25, 18.25, 16.25, 17.25, 15.25, 17.25, 15.25, 16.25, 14.25, 16.25, 14.25, 15.25, 13.25, 15.25, 13.25, 14.25, 12.25, 14.25, 12.25, 13.25, 11.25, 13.25, 11.25, 12.25, 10.25, 12.25, 10.25, 11.25, 9.25, 11.25, 9.25, 10.25, 8.25, 10.25, 8.25, 9.25, 7.25, 9.25, 7.25, 8.25, 6.25, 8.25, 6.25, 7.25, 5.25, 7.25, 5.25, 6.25, 4.25, 6.25, 4.25, 5.25, 3.25, 5.25, 3.25, 4.25, 2.25, 4.25, 2.25, 3.25, 1.25, 3.25, 1.25, 2.25, 0.25, 2.25, 0.25, 1.25, 0.25, 0.25};

//    if (result_dbl != irreg_poly_area_sse_double_thrd(temp_double, sizeof(temp_double)/16))
//            printf("test failed got %f!\n", result_dbl);

////    if ((result_flt = irreg_poly_area_avx_float((float (*)[2])&temp, sizeof(temp)/8)) != 64.0f)
////        printf("test failed got %f!\n", result_flt);

////    if ((result_dbl = irreg_poly_area_avx_double((double (*)[2])&temp_double, sizeof(temp_double)/16)) != 64.0)
////        printf("test failed got %f!\n", result_dbl);

//    return 0;
//}
