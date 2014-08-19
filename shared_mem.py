__author__ = 'samyvilar'

import sys
from collections import Iterable
from itertools import repeat, izip, imap, chain
from multiprocessing import cpu_count, sharedctypes, heap
import ctypes

curr_module = sys.modules[__name__]


def product(values):  # equivalent to numpy.product, @@ product() == numpy.product() == 1.0
    values = iter(values if isinstance(values, Iterable) else repeat(values, 1))
    initial_value = next(values, 1.0)
    return reduce(lambda a, b: a * b, values, initial_value)


def shape(values):  # equivalent to numpy.shape, though it doesn't check for non-symmetrical
    lengths = []
    while hasattr(values, '__len__'):
        lengths.append(len(values))
        values = values[0]
    return tuple(lengths)


type_info = {
    'int64': {
        'aliases': {int, 'int', 'int64', 'i', 'i8', ctypes.c_int, ctypes.c_int32, long, 'l', 'q'},
        'ctype': ctypes.c_long,
        'size': 8,
    },
    'float64': {
        'aliases': {float, 'float64', ctypes.c_double, 'd'},
        'ctype': ctypes.c_double,
        'size': 8
    },
    'float32': {
        'aliases': {'float32', ctypes.c_float, 'f'},
        'ctype': ctypes.c_float,
        'size': 8
    }
}

type_info_from_aliases = dict(
    (_alias_name, dict(chain(info.iteritems(), (('name', _name),))))
    for _name, info in type_info.iteritems() for _alias_name in info['aliases']
)


def get_type_info(alias_name):
    return type_info_from_aliases[alias_name]


def ctype(type_name):
    return get_type_info(type_name)['ctype']


class _dtype(object):
    def __init__(self, name, item_size):
        self.type_name = name
        self.item_size = item_size

    def __eq__(self, other):
        return self.type_name == other

    def __str__(self):
        return 'dtype({0})'.format(self.type_name)


global_type_names = 'int64', 'float64', 'float32'
for _name in global_type_names:
    _obj = _dtype(_name, type_info[_name]['size'])
    setattr(curr_module, _name, _obj)
    type_info_from_aliases[_obj] = type_info[_name]


def dtype(type_name):  # equivalent to numpy.dtype
    return getattr(curr_module, get_type_info(type_name)['name'])


# def flat(seqs, ignore=(int, long, float, basestring)):  # gets the base non iterable element ...
#     return repeat(seqs, 1) if any(imap(isinstance, repeat(seqs), ignore)) else chain.from_iterable(imap(flat, seqs))


def flat(seqs, ignore={int, long, float, str, unicode}):  # gets the base non iterable element ...
    return repeat(seqs, 1) if type(seqs) in ignore or not isinstance(seqs, Iterable) else chain.from_iterable(imap(flat, seqs))


class Ctypes(object):
    def __init__(self, data):
        self.data = data

    def data_as(self, ctype):
        return ctypes.cast(self.data, ctype)


class Array(object):
    def __init__(self, _buffer, _flatten_values, _dtype, _shape):
        data = ctypes.addressof(_flatten_values)
        self.buffer = _buffer
        # self.flat_values = _flatten_values
        self.dtype = _dtype
        self.shape = _shape
        self.ctypes = Ctypes(data)
        self.c_values = product(chain((ctype(_dtype),), reversed(_shape))).from_address(data)
        self.itemsize = _dtype.item_size
        self.size = product(_shape)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, item):
        value = self.c_values[item]
        if isinstance(item, slice):
            return [Array(self.buffer, segment, self.dtype, shape(segment)) for segment in value]
        if hasattr(value, '__len__'):  # TODO: better(safer) approach ...
            return Array(self.buffer, self.c_values[item], self.dtype, shape(self.c_values[item]))
        return value


def allocate_aligned_shared_mem_c_array(
    shape_or_count=(),
    elem_type=None,
    alignment=32,
    segment_count=cpu_count(),
    init=()
):
    shape_or_count = shape_or_count or shape(init)
    item_count = product(shape_or_count)
    item_type = dtype(elem_type or next(imap(type, flat(init)), elem_type))
    item_size = item_type.item_size
    array_size = ((item_size * item_count) + (segment_count * alignment))

    wrapper = heap.BufferWrapper(array_size)
    raw_array = sharedctypes.rebuild_ctype(ctypes.c_byte * array_size, wrapper, None)
    aligned_address = ctypes.addressof(raw_array)
    aligned_address += aligned_address % max(alignment, 1)
    flatten_values = (ctype(item_type) * product(shape(init))).from_address(aligned_address)

    if init:
        flatten_values[:] = tuple(flat(init))  # tuple(imap(ctype(item_type), flat(init)))

    return Array(raw_array, flatten_values, item_type, shape_or_count)


def segment(values, count=cpu_count(), alignment=32):
    # segments values, returns starting address and length of each segment, properly aligned ...
    # segments = numpy.array_split(values, cpu_count())
    # assert not values.ctypes.data % alignment

    if len(values) < count:
        return [(values.ctypes.data, len(values))] + [(0, 0)] * (count - 1)

    if len(values) == count:
        return [(entry.ctypes.data, len(entry)) for entry in values]

    indices = map(int(len(values)/count).__mul__, xrange(count))
    segments = [[values[index].ctypes.data, length] for index, length in izip(indices, repeat(len(values)/count))]
    segments[-1][1] += len(values) % count  # add any remainders ..

    for index, segment_info in enumerate(segments[1:], 1):
        remainder = segment_info[0] % alignment
        # add an element from the previous segment to the unaligned current one ...
        segment_info[0] -= remainder * values.itemsize * values[0].size  # add an element ...
        segment_info[1] += remainder
        segments[index - 1][1] -= remainder  # remove an element from previous segment ...

    assert sum(size for _, size in segments) == len(values)
    assert not any(imap(alignment.__rmod__, next(izip(*segments))))

    return segments


