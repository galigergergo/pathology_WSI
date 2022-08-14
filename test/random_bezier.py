import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('../')

from augmentation.random_bezier import get_random_points, get_bezier_curve


fig, ax = plt.subplots()
ax.set_aspect("equal")

rad = 0.2
edgy = 0.05

for c in np.array([[0, 0], [0, 1], [1, 0], [1, 1]]):

    a = get_random_points(n=7, scale=1) + c
    x, y, _ = get_bezier_curve(a, rad=rad, edgy=edgy)
    plt.plot(x, y)

plt.show()
