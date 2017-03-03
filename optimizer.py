#!/usr/bin/env python
# -*- coding: utf8 -*-
# @Time     :  10:42 AM
# @Author   : Xingdong Li
# @File     : optimizer,py

import math
import numpy as np
from scipy.integrate import simps
from numpy import sin, cos, arctan, pi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm

HEADING_ERROR_THRESHOLD = 20
SCALE_MULTIPLIER = 1.5
SCALE_PENALTY = 0.5
MAX_SCALE = 1.0
SPLIT_NUMBER = 100
DEFAULT_INITIAL_SCALE = 0.10
DEFAULT_UPDATE_TIMES = 20

# Suppose the current position is given by a 4-size array
def fit_poly(pos1, pos2, scale=DEFAULT_INITIAL_SCALE, update_times=DEFAULT_UPDATE_TIMES):
    """
    :param pos1: the inital position of the car
    :param pos2: the final position of the car:
    :param scale: the initial learning rate
    :param update_times
    :return param: the parameters of the trajectory (b, c, d) and the length of the curve
            maybe also indicator of curvature satisfaction later
    """
    k1 = pos1[3]
    k2 = pos2[3]

    # the final position of the car in new coordinator
    pos_f = transfer(pos1, pos2)
    # generate initial parameters
    param = param_init(pos_f, k1, k2)

    g = cal_error(param, pos_f, k1, k2)

    for i in range(update_times):
        param, g, scale = update(param, pos_f, k1, k2, scale, g)
        # if np.sum(np.absolute(g)) < 0.1:
        #     break
        if abs(g.item(2)) > HEADING_ERROR_THRESHOLD:
            print "pos1"
            print pos1
            print "pos2"
            print pos2
            raise Exception('heading error is not valid, polyfit failed')
    # curv_flag = curv_detect(param, k1, max_curv)

    return list(param)


def transfer(pos1, pos2):
    """
    transfer and rotate the coordinator
    :param pos1: the initial position of the car in original coordinator
    :param pos2: the final position of the  car in original coordinator
    :return: the final position of the car in rotated coordinator
    """
    trans = [pos2[0] - pos1[0], pos2[1] - pos1[1], pos2[2] - pos1[2]]
    if trans[0] == 0:
        if trans[1] > 0:
            angle = pi / 2
        else:
            angle = -pi / 2
    else:
        angle = arctan(float(trans[1]) / trans[0])
        if trans[0] < 0:
            angle += pi
    d = math.sqrt(trans[0]**2 + trans[1]**2)

    if trans[2] > np.pi:
        trans[2] -= 2 * np.pi
    if trans[2] < -np.pi:
        trans[2] += 2 * np.pi

    return [d*cos(angle-pos1[2]), d*sin(angle-pos1[2]), trans[2]]


def param_init(pos, k1, k2):
    """
    Initialize the parameters by approximation method
    :param pos: the final position of the car
    :param k1: the initial curvature of the car
    :param k2: the final curvature of the car
    :return: [b, c, d, sf]
    """
    d = math.sqrt(pos[0]**2 + pos[1]**2)
    s = d * (pos[2] ** 2 / 5.0 + 1) + 2 * abs(pos[2]) / 5.0
    b = 6.0 * pos[2] / (s ** 2) - 2 * k1 / s + 4.0 * k2 / s
    c = 3.0 * (k1 + k2) / (s ** 2) + 6.0 * pos[2] / (s ** 3)
    return [b, c, 0.0, s]


def update(param, pos_f, k1, k2, scale, g):
    """
    one turn of updating the parameters
    :param param: paramters of the curve
    :param pos_f: final position
    :param k1: initial curvature
    :param k2: final curvature
    :param scale: scale factor for this turn
    :param g: error vector in the last turn
    :return: the updated parameters
    """
    x = np.arange(0.0, param[3], float(param[3]) / SPLIT_NUMBER)
    cos_x = cos(cal_theta(x, param, k1))
    sin_x = sin(cal_theta(x, param, k1))

    # variables will be reused in the calculation
    x2 = np.power(x, 2)
    x3 = np.power(x, 3)
    x4 = np.power(x, 4)

    sf = param[3]
    sf2 = sf * sf
    sf3 = sf2 * sf
    sf4 = sf2 * sf2

    # pg = [px; py; pt; pk] is the differential of the constraints of parameters
    px = np.array([-simps(np.multiply(sin_x, x2), x) / 2.0, -simps(np.multiply(sin_x, x3), x) / 3.0,
                   -simps(np.multiply(sin_x, x4), x) / 4.0, cos_x[len(cos_x)-1]])
    py = np.array([simps(np.multiply(cos_x, x2), x) / 2.0, simps(np.multiply(cos_x, x3), x) / 3.0,
                   simps(np.multiply(cos_x, x4), x) / 4.0, sin_x[len(sin_x)-1]])

    pt = np.array([sf2 / 2.0, sf3 / 3.0, sf4 / 4.0, cal_k(sf, param, k1)])
    pk = np.array([sf, sf2, sf3, param[0] + 2 * param[1] * sf +
                   3 * param[2] * sf2])

    pg = np.matrix([px, py, pt, pk])

    # g is the error among the current final position and the constraints

    delta = np.linalg.inv(pg) * g.transpose()

    for i in range(len(param)):
        param[i] += delta.item(i) * min(scale * SCALE_MULTIPLIER, MAX_SCALE)

    g_new = cal_error(param, pos_f, k1, k2)
    if scale < MAX_SCALE and (abs(g_new.item(0)) > abs(g.item(0)) or abs(g_new.item(1)) > abs(g.item(1)) or
                      abs(g_new.item(2)) > abs(g.item(2))):
        for i in range(len(param)):
            param[i] -= delta.item(i) * min(scale * SCALE_PENALTY, MAX_SCALE - scale)
        g_new = cal_error(param, pos_f, k1, k2)
    else:
        scale = min(scale*SCALE_MULTIPLIER, MAX_SCALE)

    return param, g_new, scale


def cal_error(param, pos_f, k1, k2):
    """
    calculate the error between the required position and the position defined by current parameters
    :param param: parameters of the curve
    :param pos_f: the required position
    :param k1: initial curvature
    :param k2: final curvature
    :return: error array (x_error, y_error, theta_error, k_error)
    """
    x = np.arange(0, param[3], float(param[3]) / SPLIT_NUMBER)

    cos_x = cos(cal_theta(x, param, k1))
    sin_x = sin(cal_theta(x, param, k1))

    return np.matrix([pos_f[0] - simps(cos_x, x),
                      pos_f[1] - simps(sin_x, x),
                      pos_f[2] - cal_theta(param[3], param, k1),
                      k2 - cal_k(param[3], param, k1)
                      ])


def cal_theta(s, param, k1):
    """
    calculate the heading after moving s distance
    :param s: the distance moved
    :param param: parameters of the curve
    :param k1: initial curvature
    :return: heading value
    """
    res = param[2] / 4.0 * s
    res = np.multiply(res + param[1] / 3.0, s)
    res = np.multiply(res + param[0] / 2.0, s)
    res = np.multiply(res + k1, s)
    return res


def cal_k(s, param, k1):
    """
    calculate the curvature after moving s distance
    :param s: the distance moved
    :param param: parameters of the curve
    :param k1: initial curvature
    :return: curvature value
    """
    res = param[2] * s
    res = np.multiply(res + param[1], s)
    res = np.multiply(res + param[0], s)
    res = res + k1
    return res


def curv_detect(param, k1, max_curv):
    """
    judge whether the curvature in this road period exceeds the bound
    :param param: parameters of the curve
    :param k1: initial curvature
    :param max_curv: curvature bound
    :return: true or false
    """
    x = np.arange(0, param[3], float(param[3])/SPLIT_NUMBER)

    k = cal_k(x, param, k1)

    if max(np.absolute(k)) <= max_curv:
        return True
    else:
        return False


def plot(pos1, pos2, step1, step2):

    k1 = pos1[3]
    k2 = pos2[3]

    # the final position of the car in new coordinator
    pos_f = transfer(pos1, pos2)
    # generate initial parameters
    param = param_init(pos_f, k1, k2)

    x = np.arange(param[0]-step1*20, param[0]+step1*10, step1)
    y = np.arange(param[1]-step2*10, param[1]+step2*10, step2)
    X, Y = np.meshgrid(x, y)
    Z = []
    for b, c in zip(np.ravel(X), np.ravel(Y)):
        temp_param = [b, c, param[2], param[3]]
        error = cal_error(temp_param, pos_f, k1, k2)
        error = error.reshape(4, 1)
        Z.append(sum(np.multiply(error, error)))
    Z = np.array(Z)
    print Z.shape
    print X.shape
    Z = Z.reshape(X.shape)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

if __name__ == '__main__':
    # fit_poly([3380.9432909671741, -1718.217079946334, -1.3960183421702634, -0.3], [3391.520113, -1734.33504, -1.0268122270006699, 0], 0.1, 20)
    plot([3380.9432909671741, -1718.217079946334, -1.3960183421702634, 0], [3391.520113, -1734.33504, -1.0268122270006699, 0], 0.005, 0.001)