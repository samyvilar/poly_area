import multiprocessing

__author__ = 'samyvilar'

from itertools import islice, izip, imap, chain, starmap, repeat, product, izip_longest
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


prefix_name, cord_type_names, intrinsic_names = 'irreg_poly_area', {'float', 'double'}, {'', 'sse', 'avx'}


def c_impl_poly_area_name(memb_type_name, intrs_name='', prefix_name=prefix_name):
    return prefix_name + '_' + (intrs_name + (intrs_name and '_')) + memb_type_name


c_impl_names = tuple(starmap(c_impl_poly_area_name, product(cord_type_names, intrinsic_names)))


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


def mac_avx_supported():
    return ' AVX1.0 ' in subprocess.check_output(('sysctl', '-a', 'machdep.cpu.features'))


def linux_avx_supported():
    return ' avx ' in subprocess.check_output(('cat', '/proc/cpuinfo'))


def win_avx_supported():
    raise NotImplementedError


default_impls = {'Darwin': mac_avx_supported, 'Linux': linux_avx_supported, 'Windows': win_avx_supported}


def avx_supported(impls=default_impls):
    return impls[platform.system()]()


def broad_cast(func, values):
    pass


if __name__ == '__main__':
    # pre_allocate all the arguments ...
    polygon = diag_gen(2*10**5)
    aligned_mem_block_float = numpy_allocate_aligned_shared_mem_block((len(polygon), 2), 'float32', 32)
    aligned_mem_block_double = numpy_allocate_aligned_shared_mem_block((len(polygon), 2), 'float64', 32)

    aligned_mem_block_float[:] = polygon
    aligned_mem_block_double[:] = polygon

    polygon_floats = aligned_mem_block_float.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    polygon_doubles = aligned_mem_block_double.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    # segment_sizes = [len(polygon)/cpu_count()] * cpu_count()
    # polygon_floats_addresses = tuple(
    #     aligned_mem_block_float[cpu_index*segment_size:((cpu_index + 1)*segment_size + 1)].ctypes.data
    #     for cpu_index, segment_size in enumerate(segment_sizes)
    # )
    # for i in xrange(len(segment_sizes) - 1):
    #     segment_sizes[i] += 2
    # segment_sizes[-1] += len(polygon) % cpu_count()
    # args = zip(polygon_floats_addresses, segment_sizes)
    #
    # def apply_irreg_poly_area_from_addrs(args):
    #     return poly_area_so.irreg_poly_area_sse_float(ctypes.cast(args[0], ctypes.POINTER(ctypes.c_float)), args[1])
    #
    # def poly_area_sse_float_multi_cpu(args, pool=multiprocessing.Pool(processes=cpu_count())):
    #     return abs(sum(pool.map(apply_irreg_poly_area_from_addrs, args)))/2

    impls =                         \
        ('area', 'polygon'),        \
        ('area_iter', 'polygon'),   \
        ('poly_area_so.' + c_impl_poly_area_name('double'), 'polygon_doubles, len(polygon)'),           \
        ('poly_area_so.' + c_impl_poly_area_name('float'), 'polygon_floats, len(polygon)'),             \
        ('poly_area_so.' + c_impl_poly_area_name('double', 'sse'), 'polygon_doubles, len(polygon)'),    \
        ('poly_area_so.' + c_impl_poly_area_name('float', 'sse'), 'polygon_floats, len(polygon)')

    if avx_supported():
        impls += (
            ('poly_area_so.' + c_impl_poly_area_name('float', 'avx'), 'polygon_floats, len(polygon)'),
            ('poly_area_so.' + c_impl_poly_area_name('double', 'avx'), 'polygon_doubles, len(polygon)')
        )

    repeat_cnt = 10
    setup = 'from __main__ import poly_area_so, area, area_iter, polygon, polygon_doubles, polygon_floats'

    base_line_result = area(polygon)
    base_line_time = timeit.timeit('area(polygon)', setup=setup, number=repeat_cnt)
    errors = (abs((base_line_result - eval(expr))/base_line_result) for expr in starmap('{0}({1})'.format, impls))
    timings = imap(
        timeit.timeit,
        starmap('{0}({1})'.format, impls),
        repeat(setup),
        repeat(timeit.default_timer),
        repeat(repeat_cnt)
    )

    for info, timing, error in izip(impls, timings, errors):
        print('{name}: {avg_time}s, {speedup_factor}x faster vs py_area, rel_err: {rel_err}%'.format(
            name=info[0], avg_time=timing/repeat_cnt, speedup_factor=base_line_time/timing, rel_err=error*100
        ))





