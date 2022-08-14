import os
from random_bezier import get_random_points, get_bezier_curve
from ray_casting import is_inside_polygon
import numpy as np
import cv2
import pickle
from matplotlib import pyplot as plt
from time import sleep
import math
import random


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
        else:
            while x[i + 1] != x[i] - 1 and x[i + 1] != x[i]:
                x.insert(i + 1, x[i] - 1)
                y.insert(i + 1, y[i])
        i += 1
    return x, y


def build_intersection_dict(x, y, N):
    dct = dict()
    for i in range(N):
        dct[i] = []
    for i in range(N):
        for j in range(len(x)):
            if x[j] == i and y[j] not in dct[i]:
                dct[i].append(y[j])
    return dct


def is_inner_point_ray_casting_vert_by_dict(p_coords, vert_dict):
    intersections = 0
    i = p_coords[1]
    while i < N:
        if i in vert_dict[p_coords[0]]:
            intersections += 1
            # does not count if they are next to each other
            while i + 1 in vert_dict[p_coords[0]]:
                i += 1
        i += 1
    if intersections % 2 == 0:
        return False
    return True


def is_inner_point_ray_casting_horiz_by_dict(p_coords, horiz_dict):
    intersections = 0
    i = p_coords[0]
    while i < N:
        if i in horiz_dict[p_coords[1]]:
            intersections += 1
            # does not count if they are next to each other
            while i + 1 in horiz_dict[p_coords[1]]:
                i += 1
        i += 1
    if intersections % 2 == 0:
        return False
    return True


def is_inner_point_ray_casting_vert(p_coords, xy, N):
    intersections = 0
    i = p_coords[0]
    while i < N:
        if (i, p_coords[1]) in xy:
            intersections += 1
            # does not count if they are next to each other
            while (i + 1, p_coords[1]) in xy:
                i += 1
        i += 1
    if intersections % 2 == 0:
        return False
    return True


def is_inner_point_ray_casting_horiz(p_coords, xy, N):
    intersections = 0
    i = p_coords[1]
    while i < N:
        if (p_coords[0], i) in xy:
            intersections += 1
            # does not count if they are next to each other
            while (p_coords[0], i + 1) in xy:
                i += 1
        i += 1
    if intersections % 2 == 0:
        return False
    return True


def count_diff_neighbours(i, j, matrix):
    inds = [(i-1, j-1), (i-1, j), (i-1, j+1), (i, j-1),
            (i, j+1), (i+1, j-1), (i+1, j), (i+1, j)]
    n, m = matrix.shape
    count = 0
    for ii, jj in inds:
        if ii < 0 or jj < 0 or ii >= n or jj >= m or\
           matrix[ii, jj] != matrix[i, j]:
            count += 1
    return count


def count_diff_neighbours_far(i, j, matrix):
    inds = [(i-2, j-2), (i-2, j-1), (i-2, j), (i-2, j+1), (i-2, j+2),
            (i-1, j-2), (i-1, j+2), (i, j-2), (i, j+2), (i+1, j-2), (i+1, j+2),
            (i+2, j-2), (i+2, j-1), (i+2, j), (i+2, j+1), (i+2, j+2)]
    n, m = matrix.shape
    count = 0
    for ii, jj in inds:
        if ii < 0 or jj < 0 or ii >= n or jj >= m or\
           matrix[ii, jj] != matrix[i, j]:
            count += 1
    return count


def count_diff_neighbours_generic(i, j, matrix, dist):
    n, m = matrix.shape
    count = 0
    for iii in range(-dist, dist + 1):
        for jjj in range(-dist, dist + 1):
            if iii == jjj == 0:
                continue
            ii = i + iii
            jj = j + jjj
            if not (ii < 0 or jj < 0 or ii >= n or jj >= m) and\
               matrix[ii, jj] != matrix[i, j]:
                count += 1
    return count


# correct possible mistakes at the end of the process by changing
# values with a high different neighbour count
def end_correction_by_neighbours(mask, thresh):
    n, m = mask.shape
    for i in range(n):
        for j in range(m):
            if count_diff_neighbours(i, j, mask) > thresh:
                mask[i, j] = abs(mask[i, j] - 1)
    return mask


# correct possible mistakes at the end of the process by changing
# values with a high different neighbour count
def end_correction_by_neighbours_far(mask, thresh):
    n, m = mask.shape
    for i in range(n):
        for j in range(m):
            if count_diff_neighbours(i, j, mask) +\
               count_diff_neighbours_far(i, j, mask) > thresh:
                mask[i, j] = abs(mask[i, j] - 1)
    return mask


# correct possible mistakes at the end of the process by changing
# values with a high different neighbour count
def end_correction_by_neighbours_generic(mask, thresh, dist):
    n, m = mask.shape
    for i in range(n):
        for j in range(m):
            if count_diff_neighbours_generic(i, j, mask, dist) > thresh:
                mask[i, j] = abs(mask[i, j] - 1)
    return mask


def get_mask_outline(mask):
    n, m = mask.shape
    outline = []
    for i in range(n):
        for j in range(m):
            if count_diff_neighbours(i, j, mask) > 1:
                outline.append((i, j))
    return outline


Ns = [64, 128, 192, 256, 512]

# to_replace = [8, 18, 22, 46, 54, 74, 77, 87, 94, 98, 108, 114, 126, 135, 136,
#               143, 158, 179, 186, 197, 199, 200, 208, 212, 223, 227, 234, 237,
#               241, 256, 264, 265, 268, 281, 295, 313, 336, 342, 354, 355, 360,
#               378, 403, 414, 418, 425, 443, 447, 448, 462, 466, 471, 474, 488,
#               518, 520, 529, 531, 532, 536, 541, 547, 550, 565, 570, 572, 577,
#               615, 631, 637, 640, 664, 673, 683, 700, 705, 741, 751, 753, 755,
#               756, 757, 764, 766, 767, 774, 797, 806, 818, 829, 867, 879, 907,
#               910, 918, 931, 936, 947, 957, 979, 988, 990, 993]
to_replace = []

for k in to_replace:
    print(k)

    accepted = False

    while not accepted:
        if os.path.isfile('./images/masks/png/%d.png' % k):
            os.remove('./images/masks/png/%d.png' % k)
        if os.path.isfile('./images/masks/pkl/%d.pkl' % k):
            os.remove('./images/masks/pkl/%d.pkl' % k)

        ind = random.randint(0, 2)
        N = Ns[ind]

        rad = 0.2
        edgy = 0.05

        a = get_random_points(n=7, scale=N-math.floor(1 / 4 * N) - 5)
        x, y, _ = get_bezier_curve(a, rad=rad, edgy=edgy)
        x = x + 15
        y = y + 15
        out = np.zeros((N, N))

        x = list(map(lambda e: int(e), x))
        y = list(map(lambda e: int(e), y))
        x, y = fill_in_gaps(x, y)

        N_cps = len(x)
        for i in range(N_cps):
            out[x, y] = 1

        vert_dict = build_intersection_dict(x, y, N)
        horiz_dict = build_intersection_dict(y, x, N)
        for i in range(N):
            for j in range(N):
                if is_inner_point_ray_casting_vert_by_dict((i, j), vert_dict) and\
                   is_inner_point_ray_casting_horiz_by_dict((i, j), horiz_dict):
                    out[i, j] = 1

        dist = 1
        thresh = 3/5 * (dist * 2 + 1)**2
        out = end_correction_by_neighbours_generic(out, thresh, dist)
        dist = 2
        thresh = 3/5 * (dist * 2 + 1)**2
        out = end_correction_by_neighbours_generic(out, thresh, dist)

        outline = get_mask_outline(out)

        cv2.imwrite('./images/masks/png/%d.png' % k, out * 255)

        with open('./images/masks/pkl/outline_%d.pkl' % k, 'wb') as handle:
            pickle.dump(outline, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('./images/masks/pkl/mask_%d.pkl' % k, 'wb') as handle:
            pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)

        plt.imshow(out)
        plt.show()

        inp = input('OK? <y/n>: ')

        if inp == 'y':
            accepted = True
