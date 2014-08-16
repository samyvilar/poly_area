__author__ = 'samyvilar'

import timeit
import ctypes

import multiprocessing
from itertools import izip, repeat, imap

import poly_area
import c_poly_area
import shared_mem


def run_suites(test_size=2*10**5, repeat_cnt=10, segment_count=multiprocessing.cpu_count()):
    setup_py_data = '''
import poly_area
import c_poly_area
import shared_mem

diag_gen = poly_area.diag_gen

polygon = diag_gen({test_size})
    '''

    c_base_setup = '''
import numpy
import ctypes
import poly_area
import c_poly_area
import shared_mem

diag_gen = poly_area.diag_gen

    '''.format(','.join(c_poly_area.multi_cpu_from_segments_impl_names), ','.join(c_poly_area.c_impl_names))

    setup_c_doubles_data = c_base_setup + '''
aligned_mem_block_double = shared_mem.numpy_allocate_aligned_shared_mem_block(
    {shape}, 'float64', 32, init=diag_gen({test_size})
)
polygon_doubles = aligned_mem_block_double.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    '''

    setup_c_base_multi_cpu = c_base_setup + '''
import multiprocessing
pool = multiprocessing.Pool(processes={segment_count})
'''

    setup_c_doubles_multi_cpu_data = setup_c_doubles_data + setup_c_base_multi_cpu + '''
multi_cpu_double_args = shared_mem.segment(aligned_mem_block_double, count={segment_count})
'''

    setup_c_floats_data = c_base_setup + '''
aligned_mem_block_float = shared_mem.numpy_allocate_aligned_shared_mem_block(
    {shape}, 'float32', 32, init=diag_gen({test_size})
)
polygon_floats = aligned_mem_block_float.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
'''

    setup_c_floats_multi_cpu_data = setup_c_floats_data + setup_c_base_multi_cpu + '''
multi_cpu_float_args = shared_mem.segment(aligned_mem_block_float, count={segment_count})
'''

    polygon = poly_area.diag_gen(test_size)
    shape = (len(polygon), 2)
    setup_py_data = setup_py_data.format(test_size=test_size)
    setup_c_doubles_data = setup_c_doubles_data.format(test_size=test_size, shape=shape)
    setup_c_doubles_multi_cpu_data = setup_c_doubles_multi_cpu_data.format(
        test_size=test_size, segment_count=segment_count, shape=str(shape)
    )
    setup_c_floats_data = setup_c_floats_data.format(test_size=test_size, shape=shape)
    setup_c_floats_multi_cpu_data = setup_c_floats_multi_cpu_data.format(
        test_size=test_size, segment_count=segment_count, shape=str(shape)
    )

    c_impl_multi_cpu = map('c_poly_area.'.__add__, c_poly_area.multi_cpu_from_segments_impl_names)
    c_impl_names = map('c_poly_area.'.__add__, c_poly_area.c_impl_names)

    c_double_impl_names = [n for n in c_impl_names if 'double' in n]
    c_float_impl_names = [n for n in c_impl_names if 'float' in n]

    benchmark_suites = (
        ('python',
         setup_py_data,
         tuple(izip(imap('poly_area.'.__add__, poly_area.python_impl_names), repeat('polygon')))),

        ('c_double_impls',
         setup_c_doubles_data,
         tuple(izip(c_double_impl_names, repeat('polygon_doubles, {0}'.format(len(polygon)))))),

        ('c_float_impls', setup_c_floats_data,
            tuple(izip(c_float_impl_names, repeat('polygon_floats, {0}'.format(len(polygon)))))),

        ('c_double_impls_multi_cpu', setup_c_doubles_multi_cpu_data,
            tuple(izip((n + '_multi_cpu_from_ptrs' for n in c_double_impl_names),
                       repeat('multi_cpu_double_args, pool=pool')))),

        ('c_float_impls_multi_cpu', setup_c_floats_multi_cpu_data,
            tuple(izip((n + '_multi_cpu_from_ptrs' for n in c_float_impl_names),
                       repeat('multi_cpu_float_args, pool=pool'))))
    )

    base_line_result = poly_area.area(poly_area.diag_gen(test_size))
    base_line_time = timeit.timeit('poly_area.area(polygon)', setup=setup_py_data, number=repeat_cnt)

    aligned_mem_block_float = shared_mem.numpy_allocate_aligned_shared_mem_block(shape, 'float32', 32, init=polygon)
    polygon_floats = aligned_mem_block_float.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    aligned_mem_block_double = shared_mem.numpy_allocate_aligned_shared_mem_block(shape, 'float64', 32, init=polygon)
    polygon_doubles = aligned_mem_block_double.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    multi_cpu_float_args = shared_mem.segment(aligned_mem_block_float)
    multi_cpu_double_args = shared_mem.segment(aligned_mem_block_double)
    for index in xrange(len(multi_cpu_float_args) - 1):
        multi_cpu_float_args[index][1] += 1
        multi_cpu_double_args[index][1] += 1

    pool = multiprocessing.Pool(processes=segment_count)

    for suite_name, setup, impls in benchmark_suites:
        for func_name, args_str in impls:
            stmnt = '{0}({1})'.format(func_name, args_str)
            rel_error = abs((base_line_result - eval(stmnt))/base_line_result)
            timing = timeit.timeit(stmnt, setup=setup, number=repeat_cnt)
            print('{name}: {avg_time}s, {speedup_factor}x faster vs py_area, rel_err: {rel_err}%'.format(
                name=func_name,
                avg_time=timing/repeat_cnt,
                speedup_factor=base_line_time/timing,
                rel_err=rel_error * 100
            ))

if __name__ == '__main__':
    run_suites()