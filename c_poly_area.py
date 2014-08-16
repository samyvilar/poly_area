__author__ = 'samyvilar'

import multiprocessing
import logging
import ctypes
import sys

from itertools import starmap, product, izip, repeat, ifilter
from hardware_supp import supported_intrinsics
from shared_mem import numpy_allocate_aligned_shared_mem_block, segment

current_module = sys.modules[__name__]
prefix_name, cord_type_names = 'irreg_poly_area', ('float', 'double')


def c_impl_poly_area_name(memb_type_name, intrs_name='', prefix_name=prefix_name):
    return prefix_name + '_' + (intrs_name + (intrs_name and '_')) + memb_type_name


def c_func_wrapper(c_impl):
    def call_c_with_ptrs(ptr, data_len, c_impl=c_impl):
        return abs(c_impl(ptr, data_len))/2
    return call_c_with_ptrs


def py_func_wrapper_c(impl_func, memb_type, alignment=1):
    def py_func(py_cords, memb_type=memb_type, alignment=alignment, c_func=c_func_wrapper(impl_func)):
        points = numpy_allocate_aligned_shared_mem_block(
            (len(py_cords), 2),
            memb_type,
            alignment=alignment,
            init=py_cords
        )
        return c_func(points.ctypes.data_as(impl_func.argtypes[0]), len(points))
    return py_func


c_impl_names = []
py_to_c_impl_names = []

try:
    poly_area_so = ctypes.CDLL('libpoly_area.so')

    for _c_name in starmap(c_impl_poly_area_name, product(cord_type_names, supported_intrinsics)):
        try:
            c_impl_ref = getattr(poly_area_so, _c_name)
            c_intrs_type_name, c_cord_type_name = _c_name.split(prefix_name)[1].split('_')[-2:]
            c_cord_type = getattr(ctypes, 'c_' + c_cord_type_name)
            c_impl_ref.restype, c_impl_ref.argtypes = c_cord_type, [ctypes.POINTER(c_cord_type), ctypes.c_ulong]
            py_to_c_name = 'area_' + (c_intrs_type_name + (c_intrs_type_name and '_')) + c_cord_type_name + '_c'
            setattr(
                current_module,
                py_to_c_name,
                py_func_wrapper_c(c_impl_ref, c_cord_type, alignment=32 * bool(c_intrs_type_name))
            )
            setattr(current_module, _c_name, c_func_wrapper(c_impl_ref))

            py_to_c_impl_names.append(py_to_c_name)
            c_impl_names.append(_c_name)
        except Exception as er:
            logging.warning('failed to register c implementation: {0} err: {1}'.format(_c_name, er))
except OSError as _:
    logging.warning('Failed to load shared object err: {0}'.format(_))


multi_cpu_from_segments_impl_names = []
py_to_multi_cpu_from_impl_names = []
# default_pool = multiprocessing.Pool()


def apply_irreg_poly_area_from_ptrs(args):
    c_impl_name, args = args
    data_p, data_len = args
    c_impl_ref = getattr(poly_area_so, c_impl_name)
    return c_impl_ref(ctypes.cast(data_p, c_impl_ref.argtypes[0]), data_len)


for cords_type, intrinsic_type in ifilter(
    lambda args: c_impl_poly_area_name(*args) in c_impl_names,
    product(cord_type_names, supported_intrinsics)
):
    _c_name = c_impl_poly_area_name(cords_type, intrinsic_type)

    def c_multi_cpu_func_from_ptrs(segment_address_and_lens, pool=None, c_name=_c_name):
        # return sum(map(apply_irreg_poly_area_from_addrs, izip(repeat(c_impl), segment_address_and_lens)))
        return abs(sum((pool or multiprocessing.Pool()).map(
            apply_irreg_poly_area_from_ptrs,
            izip(repeat(c_name), segment_address_and_lens)))
        )/2

    def py_to_c_multi_cpu_func(
            cords, cords_type=cords_type, c_multi_cpu_func=c_multi_cpu_func_from_ptrs
    ):
        values_at_shared_mem = numpy_allocate_aligned_shared_mem_block(
            (len(cords), 2),
            dtype={'float': 'float32', 'double': 'float64'}[cords_type],
            alignment=32,
            init=cords
        )
        segments = segment(values_at_shared_mem)
        for i in xrange(len(segments) - 1):  # we need to increase the segment len by 1 for all but the very last segmnt
            segments[i][1] += 1              # when broadcasting since each impl, ignores the very last element
        return c_multi_cpu_func(segments)

    py_func_from_ptrs_name = c_impl_poly_area_name(cords_type, intrinsic_type) + '_multi_cpu_from_ptrs'

    py_func_name = c_impl_poly_area_name(cords_type, intrinsic_type) + '_multi_cpu'

    multi_cpu_from_segments_impl_names.append(py_func_from_ptrs_name)
    py_to_multi_cpu_from_impl_names.append(py_func_name)

    setattr(current_module, py_func_from_ptrs_name, c_multi_cpu_func_from_ptrs)
    setattr(current_module, py_func_name, py_to_c_multi_cpu_func)



all_py_to_c_impls = py_to_multi_cpu_from_impl_names + py_to_c_impl_names