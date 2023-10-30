import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time


def build_graf(_fig, line, column, index, title, _x, _y, _z):
    ax = _fig.add_subplot(line, column, index, projection='3d')
    ax.title.set_text(title)
    ax.plot_surface(_x, _y, _z)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    return ax


# Функция Экли
def f1(_x, _y):
    return (- 20 * np.exp(- 0.2 * np.sqrt(0.5 * (_x ** 2 + _y ** 2))) -
            np.exp(0.5 * (np.cos(2 * np.pi * _x) + np.cos(2 * np.pi * _y))) + np.e + 20)


def df1dx(_x, _y):
    return (2.8284 * _x * np.exp(-0.1414 * np.sqrt(_x ** 2 + _y ** 2))) / (np.sqrt(_x ** 2 + _y ** 2)) + np.pi * np.exp(
            0.5 * np.cos(2 * np.pi * _x) + 0.5 * np.cos(2 * np.pi * _y)) * np.sin(2 * np.pi * _x)


def df1dy(_x, _y):
    return (2.8284 * _y * np.exp(-0.1414 * np.sqrt(_x ** 2 + _y ** 2))) / (np.sqrt(_x ** 2 + _y ** 2)) + np.pi * np.exp(
            0.5 * np.cos(2 * np.pi * _x) + 0.5 * np.cos(2 * np.pi * _y)) * np.sin(2 * np.pi * _y)


# Функция Бута
def f2(_x, _y):
    return (_x + 2 * _y - 7) ** 2 + (2 * _x + _y - 5) ** 2


def df2dx(_x, _y):
    p = 10 * _x + 8 * _y - 34
    return p


def df2dy(_x, _y):
    p = 8 * _x + 10 * _y - 38
    return p


#Градиентный спуск для функции Бута
def gradient_descent_booth(x_start, y_start, a, points_arr, counter):
    for i in range(counter):
        x_new = x_start - a * df2dx(x_start, y_start)
        y_new = y_start - a * df2dy(x_start, y_start)
        points_arr.append((x_new, y_new))
        x_start = x_new
        y_start = y_new
    return points_arr
    # _ax.scatter(x_new, y_new, z_new, color='r')


#Градиентный спуск для функции Экли
def gradient_descent_eckley(x_start, y_start, a, points_arr, counter):
    for i in range(counter):
        x_new = x_start - a * df1dx(x_start, y_start)
        y_new = y_start - a * df1dy(x_start, y_start)
        points_arr.append((x_new, y_new))
        x_start = x_new
        y_start = y_new
    return points_arr


matplotlib.use("TkAgg")
fig = plt.figure()

#Данные для построение графиков
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
x, y = np.meshgrid(x, y)
z1 = f1(x, y)
z2 = f2(x, y)

#Построить граффик
ax1 = build_graf(fig, 1, 2, 1, 'Eckley Function', x, y, z1)
ax2 = build_graf(fig, 1, 2, 2, 'Booth Function', x, y, z2)

arr = []
a = 0.1
p_list = gradient_descent_booth(4, 4, a, arr, 1000)

x1 = p_list[len(p_list) - 1][0]
y1 = p_list[len(p_list) - 1][1]

#Минимум функции Экли
n_z = np.meshgrid(x1, y1)
print(f'Функция Бута (обычный градиентный спуск): f2({x1}, {y1}) = {f2(x1, y1)}')
ax2.scatter(x1, y1, f2(x1, y1), color='r')
# for i in range(len(p_list)):
#     ax2.scatter(p_list[i][0], p_list[i][1], z1, color='r')

plt.show()
