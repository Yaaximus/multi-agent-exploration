import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.DataFrame({
    'x': [12, 20, 28, 18, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72],
    'y': [39, 36, 30, 52, 54, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24]
})

# b : blue, g : green, r : red, c : cyan, m : magenta, y : yellow, k : black, w : white

colmap = {1: 'r', 2: 'g', 3: 'b', 4:'c', 5:'m', 6:'y', 7:'k', 8:'w'}

kmeans = KMeans(n_clusters=3)
kmeans.fit(df)

labels = kmeans.predict(df)
centroids = kmeans.cluster_centers_

fig = plt.figure(figsize=(5, 5))

colors = list(map(lambda x: colmap[x+1], labels))

plt.scatter(df['x'], df['y'], color=colors, alpha=0.5, edgecolor='k')
for idx, centroid in enumerate(centroids):
    plt.scatter(*centroid, color=colmap[idx+1])
plt.xlim(0, 80)
plt.ylim(0, 80)
plt.show()

# df = pd.DataFrame({
#     'x': [12, 20, 28, 18, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72],
#     'y': [39, 36, 30, 52, 54, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24]
# })


# np.random.seed(200)
# k = 3
# # centroids[i] = [x, y]
# centroids = {
#     i+1: [np.random.randint(0, 80), np.random.randint(0, 80)]
#     for i in range(k)
# }
    
# fig = plt.figure(figsize=(5, 5))
# plt.scatter(df['x'], df['y'], color='k')
# colmap = {1: 'r', 2: 'g', 3: 'b'}
# for i in centroids.keys():
#     plt.scatter(*centroids[i], color=colmap[i])
# plt.xlim(0, 80)
# plt.ylim(0, 80)
# plt.show()

######################----HUNGARIAN ALGORITHM----#######################

import numpy as np
from scipy.optimize import linear_sum_assignment
cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
print(cost)
row_ind, col_ind = linear_sum_assignment(cost)
print(row_ind)
print(col_ind)
print(cost[row_ind, col_ind])
print(cost[row_ind, col_ind].sum())

########################################################################