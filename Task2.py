import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import imageio

Nx, Ny, Nt = 100, 100, 1000

x1, x2 = 0.0, math.pi
y1, y2 = 0.0, 2.0
t1, t2 = 0.0, 1.0

hx = abs(x2 - x1) / Nx
hy = abs(y2 - y1) / Ny
ht = abs(t2 - t1) / Nt

alphax = [[0.0 for i in range(Nx)] for j in range(Ny)]
betax = [[0.0 for i in range(Nx)] for j in range(Ny)]
pointx = [[0.0 for i in range(Nx)] for j in range(Ny)]
fx = [[0.0 for i in range(Nx)] for j in range(Ny)]

alphay = [[0.0 for i in range(Nx)] for j in range(Ny)]
betay = [[0.0 for i in range(Nx)] for j in range(Ny)]
pointy = [[0.0 for i in range(Nx)] for j in range(Ny)]
fy = [[0.0 for i in range(Nx)] for j in range(Ny)]

filenames = []

xn = np.arange(x1, x2, hx)
ym = np.arange(y1, y2, hy)

for i in range(Nx):
    for j in range(Ny):
        pointy[i][j] = math.cos(xn[i]) * math.sin(math.pi * ym[j])

fig = plt.figure()
axes = Axes3D(fig)
X, Y = np.meshgrid(xn, ym)

surf = axes.plot_surface(np.array(X), np.array(Y), np.array(pointy), cmap='viridis')
axes.set_zlim3d(-1.0, 1.0)
plt.title('Solution')
plt.xlabel('y')
plt.ylabel('x')
fname = 'dump' + str(0) + '.png'
plt.savefig(fname, bbox_inches='tight')
plt.close()

for k in range(Nt + 1):

	ax = ht / (2.0 * hx ** 2)
	bx = ht / (2.0 * hx ** 2)
	cx = 1.0 + ht / hx ** 2

	for i in range(Nx):
		for j in range(Ny):
			jm1 = j - 1 if j - 1 >= 0 else j
			jp1 = j + 1 if j + 1 < Ny else j
			fx[i][j] = 0.5 * ht / (hy ** 2) * (pointy[i][jm1] + pointy[i][jp1]) + (1.0 - ht / (hy ** 2)) * pointy[i][j]

	for j in range(Ny):
		alphax[0][j] = 0.0
		betax[0][j] = 0.0
		for i in range(Nx - 1):
			alphax[i + 1][j] = bx / (cx - alphax[i][j] * ax)
			betax[i + 1][j] = (ax * betax[i][j] + fx[i][j]) / (cx - alphax[i][j] * ax)

	for j in range(Ny):
		pointx[Nx - 1][j] = betax[Nx - 1][j] / (1.0 - alphax[Nx - 1][j])
		for i in range(Nx)[Nx - 1:0:-1]:
			pointx[i - 1][j] = alphax[i][j] * pointx[i][j] + betax[i][j]

	for j in range(Ny):
		pointx[0][j] = 0.0
		pointx[Nx - 1][j] = 0.0
	for i in range(Nx):
		pointx[i][0] = pointx[i][1]
		pointx[i][Nx - 1] = pointx[i][Nx - 2]

	ay = ht / (2.0 * hy ** 2)
	by = ht / (2.0 * hy ** 2)
	cy = 1.0 + ht / hy ** 2

	for j in range(Ny):
		for i in range(Nx):
			im1 = i - 1 if i - 1 >= 0 else i
			ip1 = i + 1 if i + 1 < Nx else i
			fy[i][j] = 0.5 * ht / (hx ** 2) * (pointx[im1][j] + pointx[ip1][j]) + (1.0 - ht / (hx ** 2)) * pointx[i][j]

	for i in range(Nx):
		alphay[i][0] = 1.0
		betay[i][0] = 0.0
		for j in range(Ny - 1):
			alphay[i][j + 1] = by / (cy - alphay[i][j] * ay)
			betay[i][j + 1] = (ay * betay[i][j] + fy[i][j]) / (cy - alphay[i][j] * ay)

	for i in range(Nx):
		pointy[i][Ny - 1] = betay[i][Ny - 1] / (1.0 - alphay[i][Ny - 1])
		for j in range(Ny)[Ny - 1:0:-1]:
			pointy[i][j - 1] = alphay[i][j] * pointy[i][j] + betay[i][j]

	for j in range(Ny):
		pointy[0][j] = 0.0
		pointy[Nx - 1][j] = 0.0
	for i in range(Nx):
		pointy[i][0] = pointy[i][1]
		pointy[i][Nx - 1] = pointy[i][Nx - 2]

	if k % 5 == 0:

		X, Y = np.meshgrid(xn, ym)
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.set_zlim3d(-1.0, 1.0)
		surf = ax.plot_surface(np.array(X), np.array(Y), np.array(pointy), cmap='viridis')
		plt.title('Solution')
		plt.xlabel('y')
		plt.ylabel('x')
		fname = 'dump' + str(k) + '.png'
		filenames += [fname]
		plt.savefig(fname, bbox_inches='tight')
		plt.close()

images = []
for filename in filenames:
	images.append(imageio.imread(filename))
imageio.mimsave('movie.gif', images)