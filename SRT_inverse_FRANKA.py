import numpy as np
from numpy import cos, sin
import matplotlib.pyplot as plt
from icecream import ic
#/ SRT构型指 SRS + 3,4轴一正一负的a参数偏移
#!  此构型分支数待定, 至少为2

pi = np.pi

# * 机器人参数的输入
d_bs, d_se, d_ew, d_wt = 333., 316., 384., 107.
len_0_bs = np.array([0, 0, d_bs])
len_3_se_0 = np.array([0, d_se, 0])
len_4_ew_0 = np.array([0, 0, d_ew])
len_7_wt = np.array([0, 0, d_wt])
offset = 82.5

#* 为了计算角度范围，将某几个矩阵设置为全局变量
A_s, B_s, C_s = [], [], []
A_w, B_w, C_w = [], [], []

# * dh参数中的alpha参数
alpha = np.array([-1., 1., 1., -1., 1., 1., 0]) * pi / 2

# * 向量对应斜对称矩阵
def skew_vector(v):
    matrix = np.array([[0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                        [-v[1], v[0], 0]])
    return matrix


def solution_theta_4(phi, choice):
    k = offset
    b = 316.
    d = 384.

    a2 = 4 * k**2 + (b-d)**2 - phi**2
    a1 = 4 * k * (b+d)
    a0 = (d+b)**2 - phi**2

    up = -a1 + choice * np.sqrt(a1**2 - 4 * a0 * a2)
    down = 2 * a2
    theta = -up/down
    # print(np.arctan(theta) * 2)
    # print(theta)
    return np.arctan(theta) * 2 

# * 关节i的旋转矩阵


def rotation_axis(theta, index_joint):
    ca = cos(alpha[index_joint - 1])
    sa = sin(alpha[index_joint - 1])
    rotation = np.array([[cos(theta), -sin(theta) * ca, sin(theta) * sa],
                         [sin(theta), cos(theta) * ca, -cos(theta) * sa],
                            [0, sa, ca]])
    return rotation

# * 参考平面关节默认角的计算(肩膀), 这是一个专门的函数，变量x和y是后续计算出来的


def init_shoulder_joint(x, y):
    alpha = np.arctan2(x[2], x[0])
    beta = np.arctan2(y[1], y[0])
    amplitude1 = np.sqrt(x[0]**2 + x[2]**2)
    amplitude2 = np.sqrt(y[0]**2 + y[1]**2)
    theta_02 = np.arcsin(- y[2] / amplitude1) + alpha
    theta_01 = np.arcsin(- x[1] / amplitude2) + beta

    #! arcsin多值，利用第三个等式筛选至只有两种可能，目前输出其中一种
    if np.cos(theta_01 - alpha) * np.cos(theta_02 - beta) < 0:
        theta_02 = np.pi - theta_02

    return theta_01, theta_02

def inverse_with_phi(x, r, phi):
    phi = np.radians(phi)
    x_0_sw = x - len_0_bs - r @ len_7_wt
    # print('x_o_sw', x_0_sw)
    lamda = np.linalg.norm(x_0_sw)
    theta_4 = solution_theta_4(lamda, -1)
    ic(theta_4)
    alpha = - theta_4 / 2
    delta_d = np.tan(alpha) * offset 
    len_3_se = len_3_se_0 + np.array([0, delta_d, 0])
    len_4_ew = len_4_ew_0 + np.array([0,  0, delta_d])

    u_0_sw = x_0_sw / np.linalg.norm(x_0_sw)

    x_for_calculation = rotation_axis(0, 3) @ (len_3_se + rotation_axis(theta_4, 4) @ len_4_ew)
    y_for_calculation = x_0_sw
    theta_1_ref, theta_2_ref = init_shoulder_joint(x_for_calculation, y_for_calculation)
    r_03_ref = rotation_axis(theta_1_ref, 1) @ rotation_axis(theta_2_ref, 2) @ rotation_axis(0, 3)

    cross_matrix_sw = skew_vector(u_0_sw)
    A_s = cross_matrix_sw @ r_03_ref
    B_s = - cross_matrix_sw @ cross_matrix_sw @ r_03_ref
    C_s = np.array(np.matrix(u_0_sw).T @ np.matrix(u_0_sw)) @ r_03_ref
    r_03 = A_s * sin(phi) + B_s * cos(phi) + C_s

    #! 计算theta1,2,3, 多值
    theta_1 = np.arctan2(r_03[1, 1] , r_03[0, 1])
    theta_2 = np.arccos(r_03[2, 1])
    theta_3 = np.arctan2(-r_03[2, 2] , -r_03[2, 0])

    # 腕关节角计算
    # 相应矩阵
    A_w = rotation_axis(theta_4, 4).T @ A_s.T @ r
    B_w = rotation_axis(theta_4, 4).T @ B_s.T @ r
    C_w = rotation_axis(theta_4, 4).T @ C_s.T @ r

    r_47 = A_w * sin(phi) + B_w * cos(phi) + C_w

    #! 计算theta5,6,7, 多值
    theta_5 = np.arctan2(r_47[1, 2] , r_47[0, 2])
    theta_6 = np.arccos(-r_47[2, 2])
    theta_7 = np.arctan2(-r_47[2, 1] , r_47[2, 0]) 

    return np.array([theta_1, theta_2, theta_3, theta_4, theta_5, theta_6, theta_7])

def mdh_mat(theta, d, a, alpha):
    '''分别是dh参数的a d theta alpha'''
    matrix = np.array([
        [cos(theta), -sin(theta),  0, a],
        [cos(alpha) * sin(theta),  cos(alpha) * cos(theta), -sin(alpha), -d * sin(alpha)],
        [sin(alpha) * sin(theta),  sin(alpha) * cos(theta),  cos(alpha),  d * cos(alpha)],
        [0, 0, 0, 1]
        ])
    return matrix

def fk_franka(ang):
    Joint1 = mdh_mat(ang[0], d_bs, 0, 0)
    Joint2 = mdh_mat(ang[1], 0, 0, -pi/2)
    Joint3 = mdh_mat(ang[2], d_se, 0, pi/2)
    Joint4 = mdh_mat(ang[3], 0, offset, pi/2)
    Joint5 = mdh_mat(ang[4], d_ew, -offset, -pi/2)
    Joint6 = mdh_mat(ang[5], 0, 0, pi/2) 
    Joint7 = mdh_mat(ang[6], d_wt, 0, pi/2)
    Posture = Joint1 @ Joint2 @ Joint3 @ Joint4 @ Joint5 @ Joint6 @ Joint7
    return Posture

def fk_wrist(ang):
    Joint1 = mdh_mat(ang[0], d_bs, 0, 0)
    Joint2 = mdh_mat(ang[1], 0, 0, -pi/2)
    Joint3 = mdh_mat(ang[2], d_se, 0, pi/2)
    Joint4 = mdh_mat(ang[3], 0, offset, pi/2)
    Joint5 = mdh_mat(ang[4], d_ew, -offset, -pi/2)
    Posture = Joint1 @ Joint2 @ Joint3 @ Joint4 @ Joint5 
    return Posture[0:3, 3]    

def show_error(result, phi):
    error_rotation = result[:3, :3] @ np.linalg.inv(r_07_d) - np.eye(3)
    error_translation = result[:3, 3] - x_07_d
    print('degree {} is solvable'.format(phi))
    print('error of r: ', np.linalg.norm(error_rotation))
    print('error of x: ', np.linalg.norm(error_translation))

def fk_map(ang):
    alpha = - ang[3] / 2
    delta_d = np.tan(alpha) * offset  
    print('dd', delta_d)                                                              
                                                                                                                        
    Joint1 = mdh_mat(ang[0], 333, 0, 0)
    Joint2 = mdh_mat(ang[1], 0, 0, -pi/2)
    Joint3 = mdh_mat(ang[2], 316 + delta_d, 0, pi/2)                                     
    Joint4 = mdh_mat(ang[3], 0, 0, pi/2)
    Joint5 = mdh_mat(ang[4], 384 + delta_d, 0, -pi/2)
    Joint6 = mdh_mat(ang[5] , 0, 0, pi/2) 
    Joint7 = mdh_mat(ang[6], 107, 0, pi/2)

    Flange = mdh_mat(0, 0, 0, 0) 
    Posture = Joint1 @ Joint2 @ Joint3 @ Joint4 @ Joint5 @ Joint6 @ Joint7 @ Flange
    return Posture

if __name__ == '__main__':
    # 测试论文结果使用的位姿
    x_07_d = np.array([500, 180, 700])
    r_07_d = np.array([[0.067, 0.933, 0.354],
                    [0.933, 0.067, -0.354],
                    [-0.354, 0.354, -0.866]])

    x = x_07_d
    r = r_07_d
    phi_set = []
    joints = []

    angle = inverse_with_phi(x, r, 10)
    print(np.degrees(angle))
    result = fk_map(angle)
    print(result)

    # for phi in range(0, 360, 10):
    #     try:
    #         angle = inverse_with_phi(x, r, phi)
    #         result = fk_franka(angle)
    #         joints.append(angle)
    #         phi_set.append(phi)
    #         show_error(result, phi)
    #     except:
    #         print('degree {} is not solvable'.format(phi))

    # joints = np.array(joints)
    # # reformulate angles
    # for i in range(7):
    #     for j in range(len(phi_set) - 1):
    #         diff = joints[j+1, i] - joints[j, i]
    #         if diff > 1.8 * pi:
    #             joints[j+1, i] -= 2 * pi
    #         elif diff < -1.8 * pi:
    #             joints[j+1, i] += 2 * pi

    # for i in range(7):
    #     if i != 3:
    #         plt.plot(phi_set, np.degrees(joints[:, i]), label='joint {0}'.format(i+1))
    # plt.legend()
    # plt.show()


