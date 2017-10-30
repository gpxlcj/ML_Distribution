#! -*- coding:utf-8 -*-
from random import randint
from sys import maxint
from math import sin, cos, log, e, sqrt, pi
from Regression import RidgeRegression
from matrix_tool import *

'''
 Problem 1(a): univariate gaussian data generate model
'''
def univariate_gaussian_data_generate(m=0, s=1, n=20):
    uniform_data_u = [randint(0, maxint)/float(maxint) for i in range(0, n)]
    uniform_data_v = [randint(0, maxint)/float(maxint) for i in range(0, n)]

    data = list()
    for i in range(0, n):
        u = uniform_data_u[i]
        v = uniform_data_v[i]
        x = sqrt(-2*log(u, e)) * cos(2*pi*v)
        x = m+s*x
        data.append(x)
    return data

'''
Problem 1(b): Polynomial Basis Linear Model Data Generate model
'''
def linear_data_generate_model(n=1, a=1, w=[1]):
    if n != len(w):
        print ('the length of w is unequal to n')
        return 0
    #generate bias e
    e = univariate_gaussian_data_generate(0, a, 1)[0]
    x = randint(0, maxint)/float(maxint) * 20 - 10
    phi_x = 1
    y = e
    x_list = list()
    for i in range(0, n):
        y = y + w[0] * phi_x
        x_list.append(phi_x)
        phi_x *= x
    print('y: '),
    print(y)
    return x_list, y

'''
Problem 2: Sequential estimate the mean and variance
'''
def estimate_m_v(m, s):

    origin_avg_x = 0
    new_avg_x = 0
    origin_s = 0
    new_s = 0
    n = 0
    while(1):
        n += 1
        x = univariate_gaussian_data_generate(m, s, 1)[0]
        new_avg_x = origin_avg_x + (x - origin_avg_x)/n
        new_s = ((n-1)*s + (x - origin_avg_x) * (x - new_avg_x))/n
        origin_avg_x = new_avg_x
        origin_s = new_s
        print(x),
        print(new_avg_x),
        print(new_s)
        #judge convergence
        if abs(origin_avg_x - new_avg_x)<0.01:
            print(n)
            break

'''
Problem 3: Baysian Linear Regression
'''
def baysian_linear_regression(b=1.0, n=1, a=1.0, g_w=[1]):

    w = [0 for i in range(n)]
    t_x_list = list()
    y_list = list()
    form_bias = 100

    while(1):

        m_I = list()
        for i in range(n):
            m_I.append(list())
            for j in range(n):
                m_I[i].append(0)
            m_I[i][i] = b
        #generate data point
        x_list, y = linear_data_generate_model(n, a, g_w)
        t_x_list.append(x_list)
        y_list.append(y)
        data_len = len(y_list)
        t_x_list_trans = zip(*t_x_list)
        multi_data_matrix = matrix_time(t_x_list_trans, t_x_list)
        for i in range(n):
            multi_data_matrix = ele_multiply(multi_data_matrix, i, a)
        s = matrix_add(m_I, multi_data_matrix)
        s_inverse = lu_decomposition(s, n)
        mean =  matrix_time(s_inverse, t_x_list_trans)
        mean =  matrix_time(mean, [[i] for i in y_list])
        for i in range(n):
            mean = ele_multiply(mean, i, a)
        print(s),
        print(mean)
        data_list = [[t_x_list[i][1], y_list[i]] for i in range(data_len)]
        reset_file = str(randint(1, 10))
        regression = RidgeRegression(n, b/a, reset_file)
        regression.update_data_list(data_list)
        regression.train()
        print (regression.x)
        # justify convergence
        bias = sum([(regression.x[i][0]-g_w[i])**2 for i in range(n)])/n
        if (form_bias - bias)<0.001:
            break
        form_bias = bias
        regression = None


if __name__ == '__main__':
    m = 3
    s = 2
    n = 1000
#p1
    data = univariate_gaussian_data_generate(m, s, n)
    print ('problem 1 ANS: ')
    print('Part A')
    print ('data number: '),
    print (data)
    print(sum(data)/n)
    print('Part B')
    linear_data_generate_model(2, 2, [2, 3])

    print ('\n')
#p2
    print ('problem 2 ANS: ')
    estimate_m_v(0, 1)

    print ('\n')
#p3
    print ('problem 3 ANS: ')
    baysian_linear_regression(1, 4, 1, [3, 2, 5, 7])

    print ('\n')