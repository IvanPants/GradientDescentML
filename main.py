import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy.utilities.lambdify import lambdify
from sympy import cos, sin, pi, E, exp
import time


def build_graf(_fig, line, column, index, title, _x, _y, _z):
        ax = _fig.add_subplot(line, column, index, projection='3d')
        ax.plot_surface(_x, _y, _z, alpha=0.25)
        ax.title.set_text(title)
        # ax.set_xlim((-15, 15))
        # ax.set_ylim((-15, 15))
        # ax.set_zlim((-15, 15))
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
    return 10 * _x + 8 * _y - 34


def df2dy(_x, _y):
    return 8 * _x + 10 * _y - 38


def f3(_x, _y):
    #user_input = input('Введите функцию : ')
    user_input = "x**2 + y**2+cos(x*y)"
    # user_input = "- 20 * exp(-0.2 * sqrt(0.5 * (x ** 2 + y ** 2))) -\
    #          exp(0.5 * (cos(2 * pi * x) + cos(2 * pi * y))) + E + 20"
    #user_input = "(x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2"
    x, y = sp.symbols('x y', real=True)
    locals = {'x': x, 'y': y}
    expr = sp.sympify(user_input, locals=locals)
    #print(f'user_input = {expr}')
    f = lambdify([x, y], expr, 'numpy')
    # expr = expr.subs('x', _x)
    # expr = expr.subs('y', _y)

    return f(_x, _y), expr


def df3dx(_x, _y):
    x, y = sp.symbols('x y', real=True)
    p = sp.diff(glob_expr, x)
    p = p.evalf(subs={y: _y})
    p = p.evalf(subs={x: _x})

    return p.evalf()


def df3dy(_x, _y):
    x, y = sp.symbols('x y', real=True)
    p = sp.diff(glob_expr, y)
    p = p.evalf(subs={x: _x})
    p = p.evalf(subs={y: _y})

    return p


def gradient_descent(fun1, fun2, x_start, y_start, a, counter=None):
    points_arr = []
    for i in range(counter):
        x_new = x_start - a * fun1(x_start, y_start)
        y_new = y_start - a * fun2(x_start, y_start)
        points_arr.append((x_new, y_new))
        x_start = x_new
        y_start = y_new

    return points_arr


def instant_gradient_descent():
    pass


def animate_grad_descent(arr):
    pass


# def apply_gradient_descent(arr, title, ax, fun=None):
#     x, y = sp.symbols('x y', real=True)
#     x_start = arr[0][0]
#     y_start = arr[1][1]
#     ax.scatter3D(x_start, y_start, fun(x_start, y_start), s=10, ec='green', marker='X')
#     for i in range(1, len(arr)-2):
#         f = expr
#         _x = arr[i][0]
#         _y = arr[i][1]
#         f = f.evalf(subs={x: _x})
#         f = f.evalf(subs={y: _y})
#         ax.scatter3D(_x, _y, fun(_x, _y), s=10, ec='black', marker='v')
#     ax.scatter3D(arr[len(arr)-1][0], arr[len(arr)-1][0], fun(arr[len(arr)-1][0], arr[len(arr)-1][0]), s=10, ec='red', marker='D')
#     if expr == None:
#         print(f'{title} (классический градиентный спуск) : f({_x},{_y})= {fun(_x, _y)}')
#         # ax.scatter(_x, _y, fun(_x, _y), color='r')
#     else:
#         # x, y = sp.symbols('x y', real=True)
#         # f = expr
#         # f = f.evalf(subs={x: _x})
#         # f = f.evalf(subs={y: _y})
#         print(f'{title} (классический градиентный спуск) : f({_x},{_y})= {f}')
#         # ax.scatter(_x, _y, f, color='r')


matplotlib.use("TkAgg")
fig = plt.figure()

#Данные для построение графиков
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
x, y = np.meshgrid(x, y)
z1 = f1(x, y)
z2 = f2(x, y)
z3, glob_expr = f3(x, y)

print('user_input', glob_expr)

#Построить граффик
ax1 = build_graf(fig, 1, 3, 1, 'Eckley Function', x, y, z1)
ax2 = build_graf(fig, 1, 3, 2, 'Booth Function', x, y, z2)
ax3 = build_graf(fig, 1, 3, 3, 'User Function', x, y, z3)

#Классический градиентный спуск
a = 0.1
e_list = gradient_descent(df1dx, df1dy, 2, 2, a,  1000)
b_list = gradient_descent(df2dx, df2dy, 4, 4, a, 1000)
u_list = gradient_descent(df3dx, df3dy, 4, 4, a, 1000)
print(e_list)

# Нарисовать минимум на ргаффике
# apply_gradient_descent(e_list, 'Функция Экли', ax1, f1)
# apply_gradient_descent(b_list, 'Функция Бута', ax2, f2)
# apply_gradient_descent(u_list, 'Функция Пользователя', ax3, expr)


plt.show()

