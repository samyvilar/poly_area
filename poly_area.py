import multiprocessing

__author__ = 'samyvilar'

from itertools import islice, izip, imap, chain, starmap, repeat, product, izip_longest, ifilterfalse, ifilter
import logging
import ctypes
import numpy
import timeit
import sys
import platform
import subprocess

from multiprocessing import sharedctypes, cpu_count

current_module = sys.modules[__name__]


def diag_gen(length):  # 1 loops, best of 3: 543 ms per loop
    points = [(0.25, 0.25)]
    for _ in range(1, length + 1):
        points.append((points[-1][0] + 1, points[-1][1]))
        points.append((points[-1][0], points[-1][1] + 1))
    points.append((points[-1][0], points[-1][1] + 1))
    points.append((points[-1][0] - 1, points[-1][1]))
    points.append((points[-1][0], points[-1][1] - 1))
    for _ in range(1, length):
        points.append((points[-1][0] - 1, points[-1][1]))
        points.append((points[-1][0], points[-1][1] - 1))
    points.append((0.25, 0.25))
    return points


def test_diag_gen(points):
    res = 0
    for point in points:
        assert abs(point[0] + point[1] - res) <= 1
        res = point[0] + point[1]
    assert points[0] == points[-1]


def area(points):
    acc = 0
    for i in range(len(points) - 1):
        acc += points[i][0] * points[i + 1][1] - points[i + 1][0] * points[i][1]
    return abs(acc) / 2


def area_iter(points):
    # area of an irregular polygon is calculated as follows:
    # 1) take the product of the current cord of x and the next cord of y
    # 2) take the product of the current cord of y and the next cord of x
    # 3) take the difference of the two products
    # 4) sum up all the differences
    # 5) take half of the magnitude of the sum
    # return abs(sum( # 1 loops, best of 3: 489 ms per loop
    #     (curr_x * next_y) - (curr_y * next_x)
    #     for curr_x, curr_y, next_x, next_y in imap(chain.from_iterable, izip(points, islice(points, 1, None)))
    # )) / 2
    return abs(sum(  # 1 loops, best of 3: 336 ms per loop
        (curr_point[0] * next_point[1]) - (curr_point[1] * next_point[0])
        for curr_point, next_point in izip(points, islice(points, 1, None))
    )) / 2


# noinspection PyNoneFunctionAssignment
def numpy_allocate_aligned_shared_mem_block(
        shape_or_count,
        dtype,
        alignment_bytes=0,
        segment_count=cpu_count(),
        init=()
):
    # Create an aligned numpy array using a shared memory block,
    # in order to avoid copying values in between processes as well be able to call faster *_load_* intrinsics ...
    item_count, item_size = numpy.product(shape_or_count), numpy.dtype(dtype).itemsize
    # Add alignment bytes for better SSE/AVX performance ...
    size_in_bytes = ((item_size * item_count) + (segment_count * alignment_bytes))
    raw_array = sharedctypes.RawArray('b', size_in_bytes)
    _buf = numpy.frombuffer(raw_array, dtype='b')
    start_index = -_buf.ctypes.data % max(alignment_bytes, 1)
    # get numpy array aligned ...
    values = _buf[start_index:start_index + item_count*item_size].view(dtype).reshape(shape_or_count)
    if init:
        values[:] = init
    return values


prefix_name, cord_type_names, intrinsic_names = 'irreg_poly_area', ('float', 'double'), ('', 'sse', 'avx')


def c_impl_poly_area_name(memb_type_name, intrs_name='', prefix_name=prefix_name):
    return prefix_name + '_' + (intrs_name + (intrs_name and '_')) + memb_type_name


def conv_to_numpy(values, dtype):
    return numpy.asarray(values, dtype=dtype) if type(values) != numpy.ndarray or values.dtype != dtype else values


def conv_to_numpy_aligned(values, dtype, alignment_bytes):
    if type(values) != numpy.ndarray or values.dtype != dtype or values.ctypes.data % alignment_bytes:
        values = numpy_allocate_aligned_shared_mem_block(
            (len(values), 2), dtype, alignment_bytes=alignment_bytes, init=values
        )
    return values


def py_func_wrapper_c(impl_func, memb_type, alignment=0):
    def py_func(points, memb_type=memb_type, alignment=alignment, impl_func=impl_func):
        points = (alignment and conv_to_numpy or conv_to_numpy)(points, memb_type)
        return impl_func(points.ravel().ctypes.data_as(impl_func.argtypes[0]), len(points))
    return py_func


def mac_avx_supported():
    return ' AVX1.0 ' in subprocess.check_output(('sysctl', '-a', 'machdep.cpu.features'))


def linux_avx_supported():
    return ' avx ' in subprocess.check_output(('cat', '/proc/cpuinfo'))


def win_avx_supported():
    raise NotImplementedError


default_impls = {'Darwin': mac_avx_supported, 'Linux': linux_avx_supported, 'Windows': win_avx_supported}


def avx_supported(impls=default_impls):
    return impls[platform.system()]()


supported_intrinsics = ('', 'sse', 'avx' * avx_supported())
c_impl_names = tuple(starmap(c_impl_poly_area_name, product(cord_type_names, supported_intrinsics)))


try:
    poly_area_so = ctypes.CDLL('libpoly_area.so')

    for _c_name in c_impl_names:
        try:
            c_impl_ref = getattr(poly_area_so, _c_name)
            c_intrs_type_name, c_cord_type_name = _c_name.split(prefix_name)[1].split('_')[-2:]
            c_cord_type = getattr(ctypes, 'c_' + c_cord_type_name)
            c_impl_ref.argtypes = [ctypes.POINTER(c_cord_type), ctypes.c_ulong]
            c_impl_ref.restype = c_cord_type
            setattr(
                current_module,
                'area_' + (c_intrs_type_name + (c_intrs_type_name and '_')) + c_intrs_type_name + '_c',
                py_func_wrapper_c(
                    c_impl_ref,
                    c_cord_type,
                    alignment=32 * bool(c_intrs_type_name)
                )
            )
        except Exception as er:
            logging.warning('failed to register c implementation: {0} err: {1}'.format(_c_name, er))

except OSError as _:
    logging.warning('Failed to load shared object msg: {0}'.format(_))



def segment(values, count=cpu_count(), alignment=32):
    # segments values, returns starting address and length of each segment, properly aligned ...
    segments = [
        [values[cpu_index*segment_size:((cpu_index + 1)*segment_size)].ctypes.data, segment_size]
        for cpu_index, segment_size in enumerate(repeat(len(values)/count, count))
    ]

    for index, segment_info in enumerate(segments[1:], 1):
        while segment_info[0] % alignment:
            segment_info[0] -= values.itemsize * values[0].size
            segment_info[1] += 1
            segments[index - 1][1] -= 1

    return segments


def apply_irreg_poly_area_from_addrs(args):
    c_impl_name = args[0]
    data_p, data_len = args[1]
    c_impl_ref = getattr(poly_area_so, c_impl_name)
    return c_impl_ref(ctypes.cast(data_p, c_impl_ref.argtypes[0]), data_len)


multi_cpu_impl_names = []

for cords_type, intrinsic_type in product(cord_type_names, supported_intrinsics):
    def c_multi_cpu_func(segment_address_and_lens, pool, c_impl=c_impl_poly_area_name(cords_type, intrinsic_type)):
        # return sum(map(apply_irreg_poly_area_from_addrs, izip(repeat(c_impl), segment_address_and_lens)))
        return abs(sum(pool.map(apply_irreg_poly_area_from_addrs, izip(repeat(c_impl), segment_address_and_lens))))/2

    def c_func(ptr, data_len, c_impl=getattr(poly_area_so, c_impl_poly_area_name(cords_type, intrinsic_type))):
        return abs(c_impl(ptr, data_len))/2

    multi_cpu_impl_names.append(c_impl_poly_area_name(cords_type, intrinsic_type) + '_multi_cpu')
    setattr(current_module, multi_cpu_impl_names[-1], c_multi_cpu_func)
    setattr(current_module, c_impl_poly_area_name(cords_type, intrinsic_type), c_func)


def run_benchmarks(test_size=2*10**5, repeat_cnt=10, segment_count=cpu_count()):
    setup_py_data = '''
from __main__ import diag_gen, area, area_iter, poly_area_so
polygon = diag_gen({test_size})
    '''

    c_base_setup = '''
from __main__ import diag_gen, numpy_allocate_aligned_shared_mem_block, poly_area_so, segment, {0}, {1}
import numpy
import ctypes
    '''.format(','.join(multi_cpu_impl_names), ','.join(c_impl_names))

    setup_c_doubles_data = c_base_setup + '''
aligned_mem_block_double = numpy_allocate_aligned_shared_mem_block({shape}, 'float64', 32, init=diag_gen({test_size}))
polygon_doubles = aligned_mem_block_double.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    '''

    setup_c_base_multi_cpu = c_base_setup + '''
import multiprocessing
pool = multiprocessing.Pool(processes={segment_count})
'''

    setup_c_doubles_multi_cpu_data = setup_c_doubles_data + setup_c_base_multi_cpu + '''
multi_cpu_double_args = segment(aligned_mem_block_double, count={segment_count})
    '''

    setup_c_floats_data = c_base_setup + '''
aligned_mem_block_float = numpy_allocate_aligned_shared_mem_block({shape}, 'float32', 32, init=diag_gen({test_size}))
polygon_floats = aligned_mem_block_float.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    '''

    setup_c_floats_multi_cpu_data = setup_c_floats_data + setup_c_base_multi_cpu + '''
multi_cpu_float_args = segment(aligned_mem_block_float, count={segment_count})
    '''

    polygon = diag_gen(test_size)
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

    python_impl_names = 'area', 'area_iter'

    c_impl_multi_cpu = tuple(
        imap('{0}_multi_cpu'.format, starmap(c_impl_poly_area_name, product(cord_type_names, supported_intrinsics)))
    )

    c_double_impl_names = [n for n in c_impl_names if 'double' in n]
    c_float_impl_names = [n for n in c_impl_names if 'float' in n]

    benchmark_suites = (
        ('python', setup_py_data, tuple(izip(python_impl_names, repeat('polygon')))),

        ('c_double_impls', setup_c_doubles_data,
            tuple(izip(c_double_impl_names, repeat('polygon_doubles, {0}'.format(len(polygon)))))),

        ('c_float_impls', setup_c_floats_data,
            tuple(izip(c_float_impl_names, repeat('polygon_floats, {0}'.format(len(polygon)))))),

        # ('c_double_impls_multi_cpu', setup_c_doubles_multi_cpu_data,
        #     tuple(izip((n + '_multi_cpu' for n in c_double_impl_names),
        #                repeat('multi_cpu_double_args, pool')))),
        #
        # ('c_float_impls_multi_cpu', setup_c_floats_multi_cpu_data,
        #     tuple(izip((n + '_multi_cpu' for n in c_float_impl_names),
        #                repeat('multi_cpu_float_args, pool'))))
    )

    base_line_result = area(diag_gen(test_size))
    base_line_time = timeit.timeit('area(polygon)', setup=setup_py_data, number=repeat_cnt)

    aligned_mem_block_float = numpy_allocate_aligned_shared_mem_block(shape, 'float32', 32, init=polygon)
    polygon_floats = aligned_mem_block_float.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    aligned_mem_block_double = numpy_allocate_aligned_shared_mem_block(shape, 'float64', 32, init=polygon)
    polygon_doubles = aligned_mem_block_double.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    multi_cpu_float_args = segment(aligned_mem_block_float)
    multi_cpu_double_args = segment(aligned_mem_block_double)
    for index in xrange(1, len(multi_cpu_float_args) - 1):
        multi_cpu_float_args[index][1] += 2
        multi_cpu_double_args[index][1] += 2

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
    run_benchmarks()




