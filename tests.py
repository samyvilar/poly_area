__author__ = 'samyvilar'

import sys
from itertools import chain, izip, repeat

import poly_area
import c_poly_area

current_module = sys.modules[__name__]


test_size = 512
expected_output = 1024


def test_diag_gen(test_size=test_size):
    points = poly_area.diag_gen(test_size)
    res = 0
    for point in points:
        assert abs(point[0] + point[1] - res) <= 1
        res = point[0] + point[1]
    assert points[0] == points[-1]


polygon = poly_area.diag_gen(test_size)

for _module, _impl_name in chain(
        izip(repeat(poly_area), poly_area.python_impl_names),
        izip(repeat(c_poly_area), c_poly_area.all_py_to_c_impls)
):
    def test_func(impl_func=getattr(_module, _impl_name)):
        got = impl_func(polygon)
        if impl_func(polygon) != expected_output:
            raise ValueError('Expected: {0} got: {1}'.format(expected_output, got))
    test_func.__name__ = 'test_' + _impl_name

    setattr(current_module, test_func.__name__, test_func)
