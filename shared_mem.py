__author__ = 'samyvilar'

from itertools import repeat, izip, imap
from multiprocessing import cpu_count, sharedctypes
import numpy


# noinspection PyNoneFunctionAssignment
def numpy_allocate_aligned_shared_mem_block(
        shape_or_count,
        dtype,
        alignment=32,
        segment_count=cpu_count(),
        init=()
):
    # Create an aligned numpy array using a shared memory block,
    # in order to avoid copying values in between processes as well be able to call faster *_load_* intrinsics ...
    item_count, item_size = numpy.product(shape_or_count), numpy.dtype(dtype).itemsize
    # Add alignment bytes for better SSE/AVX performance ...
    size_in_bytes = ((item_size * item_count) + (segment_count * alignment))
    raw_array = sharedctypes.RawArray('b', size_in_bytes)
    _buf = numpy.frombuffer(raw_array, dtype='b')
    start_index = -_buf.ctypes.data % max(alignment, 1)
    # get numpy array aligned ...
    values = _buf[start_index:start_index + item_count*item_size].view(dtype).reshape(shape_or_count)
    if init:
        values[:] = init
    return values


def segment(values, count=cpu_count(), alignment=32):
    # segments values, returns starting address and length of each segment, properly aligned ...
    # segments = numpy.array_split(values, cpu_count())
    assert not values.ctypes.data % alignment
    if len(values) < count:
        return [(values.ctypes.data, len(values))] + [(0, 0)] * (count - 1)
    elif len(values) == count:
        return [(entry.ctypes.data, len(entry)) for entry in values]

    segments = [
        [values[cpu_index*segment_size:((cpu_index + 1)*segment_size)].ctypes.data, segment_size]
        for cpu_index, segment_size in enumerate(repeat(len(values)/count, count))
    ]
    segments[-1][1] += len(values) % count  # Add any remainders

    for index, segment_info in enumerate(segments[1:], 1):
        while segment_info[0] % alignment:  # While starting address isn't aligned ..
            # add an element from the previous segment to the unaligned current one ...
            segment_info[0] -= values.itemsize * values[0].size  # add an element ...
            segment_info[1] += 1
            segments[index - 1][1] -= 1  # remove an element from

    assert sum(size for _, size in segments) == len(values)
    assert all(imap(long(alignment).__rmod__, izip(*segments)))

    return segments


