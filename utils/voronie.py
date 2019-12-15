#!usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d


points = np.random.rand(10,2)

vor = Voronoi(points)


plt.figure()

for el in points:
    plt.plot(el[0], el[1], 'g*')

for el in vor.vertices:
    plt.plot(el[0], el[1], 'b*')

fig = voronoi_plot_2d(vor)

fig = voronoi_plot_2d(vor, show_vertices=False, line_colors='orange', line_width=2, line_alpha=0.6, point_size=2)
plt.show()