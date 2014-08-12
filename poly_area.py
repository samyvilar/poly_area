__author__ = 'samyvilar'

from itertools import islice, izip, imap, chain
import logging
import ctypes
import numpy
import timeit
import time

from multiprocessing import sharedctypes


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
def numpy_allocate_aligned_shared_mem_block(shape_or_count, dtype, alignment_bytes, segment_count=1):
    count = numpy.product(shape_or_count)
    item_size = numpy.dtype(dtype).itemsize
    # Add alignment bytes for better SSE performance ...
    size_in_bytes = ((item_size * count) + (segment_count * alignment_bytes))  # add a couple bytes for alignment ...
    raw_array = sharedctypes.RawArray('b', size_in_bytes)
    _buf = numpy.frombuffer(raw_array, dtype='b')
    start_index = -_buf.ctypes.data % alignment_bytes
    # get numpy array aligned ...
    return _buf[start_index:start_index + count*item_size].view(dtype).reshape(shape_or_count)


try:
    poly_area_so = ctypes.CDLL('libpoly_area.so')
    poly_area_so.area_of_irregular_polygon_from_cords_float.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_ulong]
    poly_area_so.area_of_irregular_polygon_from_cords_float.restype = ctypes.c_float

    poly_area_so.area_of_irregular_polygon_from_cords_sse_float.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_ulong
    ]
    poly_area_so.area_of_irregular_polygon_from_cords_sse_float.restype = ctypes.c_float

    poly_area_so.area_of_irregular_polygon_from_cords_avx_float.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_ulong
    ]
    poly_area_so.area_of_irregular_polygon_from_cords_avx_float.restype = ctypes.c_float

    poly_area_so.area_of_irregular_polygon_from_cords_double.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.c_ulong
    ]
    poly_area_so.area_of_irregular_polygon_from_cords_double.restype = ctypes.c_double

    def area_float_c(points):
        cords_ptr = numpy.asarray(points, dtype='float32').ravel().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        return poly_area_so.area_of_irregular_polygon_from_cords_float(cords_ptr, len(points))

    def area_float_c_sse(points):
        aligned_mem_block = numpy_allocate_aligned_shared_mem_block((len(points), 2), 'float32', 16)
        aligned_mem_block[:] = points
        return poly_area_so.area_of_irregular_polygon_from_cords_sse_float(
            aligned_mem_block.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            len(points)
        )

    def area_float_c_avx(points):
        aligned_mem_block = numpy_allocate_aligned_shared_mem_block((len(points), 2), 'float32', 32)
        aligned_mem_block[:] = points
        return poly_area_so.area_of_irregular_polygon_from_cords_avx_float(
            aligned_mem_block.ctypes.data_as(ctypes.POINTER(ctypes.c_float), len(points))
        )

    def area_double_c(points):
        cords_ptr = numpy.asarray(points, dtype='float64').ravel().ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        return poly_area_so.area_of_irregular_polygon_from_cords_double(cords_ptr, len(points))

    # if area_double_c(diag_gen(32)) != 64:
    #     raise ValueError('Expected 64 got {0}'.format(area_double_c(diag_gen(32))))
    # if area_float_c(diag_gen(32)) != 64:
    #     raise ValueError('Expected 64 got {0}'.format(area_float_c(diag_gen(32))))
    # if area_float_c_sse(diag_gen(32)) != 64:
    #     raise ValueError('Expected 64 got {0}'.format(area_float_c_sse(diag_gen(32))))

except OSError as _:
    logging.warning('Failed to load shared object msg: {0}'.format(_))

if __name__ == '__main__':
    polygon = diag_gen(2*10**5)
    aligned_mem_block_float = numpy_allocate_aligned_shared_mem_block((len(polygon), 2), 'float32', 16)
    aligned_mem_block_float[:] = polygon
    polygon_cfloats = aligned_mem_block_float.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    aligned_mem_block_double = numpy_allocate_aligned_shared_mem_block((len(polygon), 2), 'float64', 16)
    aligned_mem_block_double[:] = polygon
    polygon_cdoubles = aligned_mem_block_double.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    implementations = (area, polygon), \
                      (area_iter, polygon), \
                      (area_float_c, aligned_mem_block_float), \
                      (area_float_c_sse, aligned_mem_block_float)

    for impl in area, area_iter:
        start = time.time()
        _ = impl(polygon)
        end = time.time()
        print '{0}_py: {1}s'.format(impl.__name__, (end - start))

    start = time.time()
    _ = poly_area_so.area_of_irregular_polygon_from_cords_double(polygon_cdoubles, len(polygon))
    end = time.time()
    print '{0}: {1}s'.format('area_c_double', (end - start))

    start = time.time()
    _ = poly_area_so.area_of_irregular_polygon_from_cords_float(polygon_cfloats, len(polygon))
    end = time.time()
    print '{0}: {1}s'.format('area_c_float', (end - start))

    start = time.time()
    _ = poly_area_so.area_of_irregular_polygon_from_cords_sse_float(polygon_cfloats, len(polygon))
    end = time.time()
    print '{0}: {1}s'.format('area_c_float_sse', (end - start))

    start = time.time()
    _ = poly_area_so.area_of_irregular_polygon_from_cords_avx_float(polygon_cfloats, len(polygon))
    end = time.time()
    print '{0}: {1}s'.format('area_c_float_avx', (end - start))





