import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy.utilities.lambdify import lambdify
from sympy import cos,sin,pi,E,exp
import time


def build_graf(_fig, line, column, index, title, _x, _y, _z):
        ax = _fig.add_subplot(line, column, index, projection='3d')
        ax.plot_surface(_x, _y, _z, alpha=0.25)
        ax.title.set_text(title)
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


def df3dx(_x, _y, expr):
    x, y = sp.symbols('x y', real=True)
    p = sp.diff(expr, x)
    p = p.evalf(subs={y: _y})
    p = p.evalf(subs={x: _x})

    return p.evalf()


def df3dy(_x, _y, expr):
    x, y = sp.symbols('x y', real=True)
    p = sp.diff(expr, y)
    p = p.evalf(subs={x: _x})
    p = p.evalf(subs={y: _y})

    return p


def gradient_descent(fun_name, x_start, y_start, a, counter, expr):
    points_arr = []
    for i in range(counter):
        if fun_name == 'e':
            x_new = x_start - a * df1dx(x_start, y_start)
            y_new = y_start - a * df1dy(x_start, y_start)
        elif fun_name == 'b':
            x_new = x_start - a * df2dx(x_start, y_start)
            y_new = y_start - a * df2dy(x_start, y_start)
        elif fun_name == 'u':
            x_new = x_start - a * df3dx(x_start, y_start, expr)
            y_new = y_start - a * df3dy(x_start, y_start, expr)
        else:
            print('Данной функции не существует')
            break
        points_arr.append((x_new, y_new))
        x_start = x_new
        y_start = y_new

    return points_arr


def apply_gradient_descent(arr, title, ax, fun, expr):
    _x = arr[len(arr)-1][0]
    _y = arr[len(arr)-1][1]
    if expr == None:
        print(f'{title} (классический градиентный спуск) : f({_x},{_y})= {fun(_x, _y)}')
        ax.scatter(_x, _y, fun(_x, _y), color='r')
    else:
        x, y = sp.symbols('x y', real=True)
        f = expr
        f = f.evalf(subs={x: _x})
        f = f.evalf(subs={y: _y})
        print(f'{title} (классический градиентный спуск) : f({_x},{_y})= {f}')
        ax.scatter(_x, _y, f, color='r')


matplotlib.use("TkAgg")
fig = plt.figure()

#Данные для построение графиков
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
x, y = np.meshgrid(x, y)
z1 = f1(x, y)
z2 = f2(x, y)
z3, expr = f3(x, y)
print('user_input', expr)

#Построить граффик
ax1 = build_graf(fig, 1, 3, 1, 'Eckley Function', x, y, z1)
ax2 = build_graf(fig, 1, 3, 2, 'Booth Function', x, y, z2)
ax3 = build_graf(fig, 1, 3, 3, 'User Function', x, y, z3)

#Классический градиентный спуск
a = 0.1
e_list = gradient_descent('e', 4, 4, a,  1000, None)
b_list = gradient_descent('b', 4, 4, a, 1000, None)
u_list = gradient_descent('u', 4, 4, a, 1000, expr)

#Нарисовать минимум на ргаффике
apply_gradient_descent(e_list, 'Функция Экли', ax1, f1, None)
apply_gradient_descent(b_list, 'Функция Бута', ax2, f2, None)
apply_gradient_descent(u_list, 'Функция Пользователя', ax3, None, expr)

plt.show()

