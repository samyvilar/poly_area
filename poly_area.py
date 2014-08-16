__author__ = 'samyvilar'

from itertools import islice, izip


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


python_impl_names = 'area', 'area_iter'
