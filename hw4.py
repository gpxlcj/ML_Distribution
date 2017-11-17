#! -*- encoding:utf-8 -*-
#author: gpxlcj


import operator
from random import randint
from sys import maxint
from math import sin, cos, log, e, sqrt, pi, floor
from matrix_tool import *
from mnist import MNIST


'''
univariate gaussian data generate model
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
Input MNIST dataset
'''

def load_mnist(dataset="training", digits=range(10), size = 60000):
    mndata = MNIST('./dataset')
    train_img, train_label = mndata.load_training()
    test_img, test_label = mndata.load_testing()
    return [[train_img, train_label], [test_img, test_label]]


'''
Problem 1: Logistic Regression
'''

class LogisticRegression:

    d1_list = list()
    d2_list = list()
    train_data = list()

    theta = [0.01, 0.01, 0.01]
    alpha = 0.01

    accuracy = 0
    sensitivity = 0
    specificity = 0
    confusion_matrix = [[0, 0], [0, 0]]
    hessian_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    xm1 = 0
    xv1 = 1
    ym1 = 0
    yv1 = 1
    xm2 = 0
    xv2 = 1
    ym2 = 0
    yv2 = 1
    n = 500
    iter = 100

    def __init__(self, xm1, xv1, ym1, yv1, xm2, xv2, ym2, yv2, n):
        temp_x = univariate_gaussian_data_generate(xm1, xv1, n)
        temp_y = univariate_gaussian_data_generate(ym1, yv1, n)
        self.d1_list = [[1, temp_x[i], temp_y[i], 1] for i in range(n)]

        temp_x = univariate_gaussian_data_generate(xm2, xv2, n)
        temp_y = univariate_gaussian_data_generate(ym2, yv2, n)
        self.d2_list = [[1, temp_x[i], temp_y[i], 0] for i in range(n)]
        self.train_data.extend(self.d1_list)
        self.train_data.extend(self.d2_list)


    def gradient_descent(self):
        n = self.n
        j_func = 0
        temp_theta = self.theta
        for i in range(self.iter):
            theta_sum = [0, 0, 0]
            for record in self.train_data:
                for j in range(3):
                    temp = e**((temp_theta[0]*record[0])+(temp_theta[1]*record[1])+(temp_theta[2]*record[2]))
                    theta_sum[j] += ((temp/(1+temp)) - record[3]) * record[j]
            for j in range(3):
                temp_theta[j] = temp_theta[j] - self.alpha * theta_sum[j]/n
        self.theta = temp_theta
        return temp_theta


    def newton_method(self):
        temp_theta = self.theta

        temp = 0
        for i in range(self.iter):
            self.hessian_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
            g_list = [[0, 0, 0]]
            for record in self.train_data:
                for j in range(3):
                    temp = e ** ((temp_theta[0]*record[0]) + (temp_theta[1]*record[1]) + (temp_theta[2]*record[2]))
                    g_list[0][j] += record[j] * (record[3] - temp/(1+temp))
                    for k in range(3):
                        self.hessian_matrix[j][k] += -record[j] * record[k] * temp/(1+temp)**2
            for j in range(3):
                g_list[0][j] = g_list[0][j] / self.n
                for k in range(3):
                    self.hessian_matrix[j][k] = self.hessian_matrix[j][k] / self.n

            self.hessian_matrix = lu_decomposition(self.hessian_matrix, 3)
            temp_matrix = matrix_time(g_list, self.hessian_matrix)[0]
            temp_matrix = [[-k for k in temp_matrix]]
            temp_theta = matrix_add([temp_theta], temp_matrix)[0]
            self.theta = temp_theta
        return temp_theta


    def evaluate(self):
        conf_m = self.confusion_matrix
        self.sensitivity = float(conf_m[1][1]) / (conf_m[1][1] + conf_m[1][0])
        self.specificity = float(conf_m[0][0]) / (conf_m[0][0] + conf_m[0][1])


    def test(self):
        self.confusion_matrix = [[0, 0], [0, 0]]
        temp_theta = self.theta
        print len(self.train_data)
        for record in self.train_data:
            temp = e ** ((temp_theta[0] * record[0]) + (temp_theta[1] * record[1]) + (temp_theta[2] * record[2]))
            if (temp/(1+temp))> 0.5:
                temp_y = 1
            else:
                temp_y = 0
            self.confusion_matrix[record[3]][temp_y] += 1


    def train(self):
        # theta_list = self.gradient_descent()
        theta_list = self.newton_method()
        print theta_list


'''
Problem 2: EM Algorithm
'''

class EMAlgo:

    train_img = list()
    train_label = list()
    test_img = list()
    test_label = list()
    p_list = list()
    sensitivity = list()
    specificity = list()
    confusion_matrix = list()
    iter = 1
    center = list()
    label = list()
    p_pi = list()
    p = list()
    cluster_assign = list()

    def __init__(self, ):
        train_group, test_group = load_mnist()
        self.train_img = list()
        self.train_label = train_group[1]
        self.test_img = test_group[0]
        self.test_label = test_group[1]
        for i in range(256):
            self.p_list.append(0)
        for i in range(len(train_group[0])):
            self.train_img.append(list())
            for j in range(256):
                self.train_img[i].append(train_group[0][i][j]/128)
                if (train_group[0][i][j]/128 == 0):
                    self.p_list[j] += 1
        for i in range(256):
            self.p_list[i] = self.p_list[i]/60000.0

        for i in range(len(test_group[0])):
            self.test_img.append(list())
            for j in range(256):
                 self.test_img[i].append(test_group[0][i][j]/128)
        print(self.train_img[0])
        for i in range(256):
            self.p.append(list())
            self.p_pi.append(list())
            for j in range(10):
                temp_a = randint(1, 100)/100.0
                self.p[i].append(temp_a)
                self.p_pi[i].append(0.1)

        for j in range(10):
            self.confusion_matrix.append([[0, 0], [0, 0]])
            self.sensitivity.append(0)
            self.specificity.append(0)


    def compare(self, a_list, b_list):
        ans = 0
        for i in range(len(a_list)):
            if a_list[i] == b_list[i]:
                ans += 1
        return ans


    def train(self):
        for i in range(256):
            print(i)
            p = self.p[i]
            p_pi = self.p_pi[i]
            for j in range(self.iter):
                #Expection step
                u = list()
                for k in range(60000):
                    u.append(list())
                    total = 0
                    for l in range(10):
                        if self.train_img[k][i] == 0:
                            temp = p_pi[l] * p[l]
                        else:
                            temp = p_pi[l] * (1-p[l])
                        u[k].append(temp)
                        total += temp
                    for l in range(10):
                        u[k][l] = u[k][l]/total
                #Maximum step
                temp_m = list()
                temp_n = list()
                for k in range(10):
                    temp_m.append(0)
                    temp_n.append(0)
                for k in range(60000):
                    for l in range(10):
                        if self.train_img[k][i] == 0:
                            temp_m[l] += u[k][l]
                        temp_n[l] += u[k][l]
                for k in range(10):
                    p_pi[k] = temp_n[k] / 60000
                    p[k] = temp_m[k] / temp_n[k]
            self.p[i] = p
            self.p_pi[i] = p_pi
        print('\n')

    def test(self):
        self.cluster_assign = list()
        for i in range(60000):
            p_min = maxint
            p_id = -1
            for j in range(10):
                total = 0
                for l in range(256):
                    if self.train_img[i][l] == 0:
                        temp = self.p[l][j]
                    else:
                        temp = 1 - self.p[l][j]
                    if temp == 0:
                        temp = 0.0000001
                    total += -log(temp, 10)
                print total,
                print j
                if total<p_min:
                    p_min = total
                    p_id = j
            self.cluster_assign.append(p_id)
        print(self.cluster_assign)
        cluster_digit = list()
        for i in range(10):
            temp_dict = dict()
            for j in range(10):
                temp_dict[str(j)] = 0
            cluster_digit.append(temp_dict)
        for i in range(60000):
            temp_name = str(self.train_label[i])
            cluster_digit[self.cluster_assign[i]][temp_name] += 1

        cluster_rep = list()
        for i in range(10):
            cluster_rep.append(max(cluster_digit[i].iteritems(), key=operator.itemgetter(1))[0])

        for i in range(10):
            pt = cluster_digit[i][cluster_rep[i]]
            pf = sum(cluster_digit[i].values()) - pt
            nt = 0
            nf = 0
            for j in range(10):
                if i != j:
                    for k in range(10):
                        if str(k)==cluster_rep[i]:
                            nf += cluster_digit[j][str(k)]
                        else:
                            nt += cluster_digit[j][str(k)]
            self.confusion_matrix[i][0][0] = nt
            self.confusion_matrix[i][0][1] = nf
            self.confusion_matrix[i][1][0] = pf
            self.confusion_matrix[i][1][1] = pt
            self.sensitivity = float(pt) / (pt+pf)
            self.specificity = float(nt) / (nt+nf)


    # def train(self):
    #     k_list = list()
    #     k_center = list()
    #     for i in range(10):
    #         k_center.append(list())
    #         k_list.append(list())
    #         for j in range(256):
    #             temp = randint(0, 100) / 100.0
    #             if (temp > self.p_list[j]):
    #                 k_center[i].append(0)
    #             else:
    #                 k_center[i].append(1)
    #
    #     temp_center_list = k_center
    #     for i in range(self.iter):
    #         print (i)
    #         k_list = list()
    #         k_label_list = list()
    #         for j in range(10):
    #             k_list.append(list())
    #             k_label_list.append(list())
    #         for j in range(60000):
    #             max_ans = 0
    #             max_label = -1
    #             for k in range(10):
    #                 temp = self.compare(self.train_img[j], k_center[k])
    #                 if (temp > max_ans):
    #                     max_ans = temp
    #                     max_label = k
    #             k_list[max_label].append(self.train_img[j])
    #             k_label_list[max_label].append(self.train_label[j])
    #         if (i == (self.iter-1)):
    #             continue
    #         k_center = list()
    #         k_label = list()
    #         for j in range(10):
    #             temp_center = list()
    #             temp_label_count = list()
    #             for m in range(256):
    #                 temp_center.append(0)
    #
    #             for k in range(len(k_list[j])):
    #                 for m in range(256):
    #                     temp_center[m] += k_list[j][k][m]
    #             if len(k_list[j]) == 0:
    #                 k_center.append(temp_center_list[j])
    #                 continue
    #             for m in range(256):
    #                 temp_center[m] = floor((float(temp_center[m]) / len(k_list[j])) / 0.5)
    #             k_center.append(temp_center)
    #         temp_center_list = k_center
    #         for j in range(10):
    #             temp_count = list()
    #             for k in range(10):
    #                 temp_count.append(0)
    #             for k in k_label_list[j]:
    #                 temp_count[k] += 1
    #             temp_max_count = 0
    #             temp_k = -1
    #             for k in range(10):
    #                 if temp_max_count<temp_count[k]:
    #                     temp_max_count = temp_count[k]
    #                     temp_k = k
    #             k_label.append(temp_k)
    #     self.center = k_center
    #     self.label = k_label
    #     print(k_label)
    #     return 0



    # def test(self):
    #     predict_list = list()
    #     for i in len(self.test_img):
    #         max_ans = 0
    #         max_label = -1
    #         for j in range(10):
    #             temp = self.compare(self.test_img[i], self.center[j])
    #             if (temp > max_ans):
    #                 max_ans = temp
    #                 max_label = j
    #         predict_list.append(max_label)


    def evaluate(self):
        conf_m = self.confusion_matrix
        self.sensitivity = float(conf_m[1][1]) / (conf_m[1][1] + conf_m[1][0])
        self.specificity = float(conf_m[0][0]) / (conf_m[0][0] + conf_m[0][1])


if __name__ == '__main__':

    #input parameter
    xm1 = 0
    xv1 = 1
    ym1 = 0
    yv1 = 1
    xm2 = 1
    xv2 = 1
    ym2 = 1
    yv2 = 1
    n = 500
    # model = LogisticRegression(xm1, xv1, ym1, yv1, xm2, xv2, ym2, yv2, n)
    # model.train()
    # model.test()
    # model.evaluate()
    # print('sensitivity: '),
    # print(model.sensitivity)
    # print('specificity: '),
    # print(model.specificity)
    # print('confusion matrix: '),
    # print(model.confusion_matrix)
    model = EMAlgo()
    model.train()
    model.test()

