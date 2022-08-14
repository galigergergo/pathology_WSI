from random_bezier import get_random_points, get_bezier_curve
from ray_casting import is_inside_polygon
import numpy as np
import cv2
from time import sleep


N = 128

rad = 0.2
edgy = 0.05

a = get_random_points(n=7, scale=N-10)
# a = list(map(lambda e: (e[0] * N, e[1] * N), a))
x, y, _ = get_bezier_curve(a, rad=rad, edgy=edgy)

out = np.zeros((N, N))

# works, but slow
# for i in range(N):
#     for j in range(N):
#         if is_inside_polygon(points=list(zip(x*N, y*N)), p=(i, j)):
#             out[i, j] = 1


# find index of the nearest row neighbour of a curve point in any direction
# neighbours are only considered, if they are at least one different index appart from the point
def find_nearest_neighbour(p_ind, p_rows):
    n_points = len(p_rows)
    count = 1
    other_between_front = False
    other_between_back = False
    while count < n_points / 2 + 1:
        if p_rows[p_ind] == p_rows[(p_ind + count) % n_points] and other_between_front:
            return (p_ind + count) % n_points
        else:
            other_between_front = True
        prev_ind = (n_points + (p_ind - count)) % n_points
        if p_rows[p_ind] == p_rows[prev_ind] and other_between_back:
            return prev_ind
        else:
            other_between_back = True
        count += 1
    return i_n


def normalize_pixel_index(ind, max_size):
    if ind < 0:
        return 0
    if ind >= max_size:
        return max_size - 1
    return ind


def fill_in_gaps(x, y):
    i = 0
    while i < len(x) - 1:
        # check whether next is larger
        growing = True
        j = 0
        while i + j < len(x) and x[i + j] == x[i]:
            j += 1
        if i + j < len(x) and x[i + j] < x[i]:
            growing = False

        # insert missing elements if existent
        if growing:
            while x[i + 1] != x[i] + 1 and x[i + 1] != x[i]:
                x.insert(i + 1, x[i] + 1)
                y.insert(i + 1, y[i])
                # print(x[:i])
                # print(x[:i+5])
                # sleep(.4)
        else:
            while x[i + 1] != x[i] - 1 and x[i + 1] != x[i]:
                x.insert(i + 1, x[i] - 1)
                y.insert(i + 1, y[i])
                # print(x[:i])
                # print(x[:i+5])
                # sleep(.4)
        i += 1
        # print(x[:i])
    return x, y


x = list(map(lambda e: int(e), x))
y = list(map(lambda e: int(e), y))
x, y = fill_in_gaps(x, y)
print(x)

N_cps = len(x)
for i in range(N_cps):
    out[x, y] = 1


for i in range(N_cps):
    i_n = find_nearest_neighbour(i, x)

    out[x[i], y[i]:y[i_n]] = 1

cv2.imwrite("polygon.png", out * 255)
