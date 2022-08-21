import pickle
import matplotlib.pyplot as plt
import numpy as np


def count_diff_neighbours(i, j, matrix):
    inds = [(i-1, j-1), (i-1, j), (i-1, j+1), (i, j-1),
            (i, j+1), (i+1, j-1), (i+1, j), (i+1, j)]
    n, m = matrix.shape
    count = 0
    for ii, jj in inds:
        if not (ii < 0 or jj < 0 or ii >= n or jj >= m) and\
           matrix[ii, jj] != matrix[i, j]:
            count += 1
    return count


def get_mask_outline(mask):
    n, m = mask.shape
    outline = []
    for i in range(n):
        for j in range(m):
            if count_diff_neighbours(i, j, mask) >= 1:
                outline.append((i, j))
    return outline


def show_outline(outline, mask):
    n = 500
    m = 500

    img = np.zeros((n, m))

    for xy in outline:
        img[xy[0], xy[1]] = 1

    plt.imshow(img)
    plt.show()

    plt.imshow(mask)
    plt.show()


for k in range(1000):
    print(k)

    with open('../../resources/masks/pkl/mask_%d.pkl' % k,
              'rb') as handle:
        mask = pickle.load(handle)

    outline = get_mask_outline(mask)

    if (0, 0) in outline:
        show_outline(outline, mask)

    with open('./images/masks/pkl/outline_%d.pkl' % k, 'wb') as handle:
        pickle.dump(outline, handle, protocol=pickle.HIGHEST_PROTOCOL)
