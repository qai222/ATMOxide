import pandas as pd
df_pred = pd.read_csv("df_cm_npred.csv", usecols=range(1,5))
df_true = pd.read_csv("df_cm_ntrue.csv", usecols=range(1,5))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

mat = np.eye(4)
mat[3,0] = 1

M = 4
N = 4
x = np.arange(M + 1)
y = np.arange(N + 1)
xs, ys = np.meshgrid(x, y)
zs_pred = np.zeros((len(xs), len(ys)))
for i in range(zs_pred.shape[0]-1):
    for j in range(zs_pred.shape[1]-1):
        zs_pred[i, j] = df_pred.values[i, j]
zs_pred = zs_pred[:-1, :-1].ravel()
zs_true = np.zeros((len(xs), len(ys)))
for i in range(zs_true.shape[0]-1):
    for j in range(zs_true.shape[1]-1):
        zs_true[i, j] = df_true.values[i, j]
zs_true = zs_true[:-1, :-1].ravel()

triangles1 = [
    (
        i + j*(M+1),  # 0 0
        i+1 + (j+1)*(M+1), # 1 1
        i + (j+1)*(M+1) # 0 1
    ) for j in range(N) for i in range(M)]
triangles2 = [
    (
        i + j*(M+1),  # 0 0
        i+1 + j*(M+1), # 1 0
        i+1 + (j+1)*(M+1), # 1 1
    ) for j in range(N) for i in range(M)]
triang1 = Triangulation(xs.ravel(), ys.ravel(), triangles1)  # top left
triang2 = Triangulation(xs.ravel(), ys.ravel(), triangles2)  # bot right
img1 = plt.tripcolor(triang1, zs_true, cmap=plt.get_cmap('copper', 100), vmax=1, vmin=0)
img2 = plt.tripcolor(triang2, zs_pred, cmap=plt.get_cmap('copper', 100), vmax=1, vmin=0)
plt.colorbar(img1, ticks=range(10))
plt.xlim(x[0], x[-1])
plt.ylim(y[0], y[-1])
plt.xticks([i+0.5 for i in range(4)], df_pred.columns)
plt.yticks([i+0.5 for i in range(4)], ["true {}".format(i) for i in range(4)])
for i in range(4):
    for j in range(4):
        c_x = i + 0.5
        c_y = j + 0.5
        top_left_center = (c_x - 0.35, c_y + 0.25)
        bot_right_center = (c_x , c_y - 0.25)
        plt.text(top_left_center[0], top_left_center[1], "{:.1%}".format(df_true.values[j, i]), c="white")
        plt.text(bot_right_center[0], bot_right_center[1], "{:.1%}".format(df_pred.values[j, i]), c="white")
plt.savefig("cm.png", dpi=300)
plt.savefig("cm.eps")


