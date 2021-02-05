import numpy as np
from numpy import cos, sin
import matplotlib.pyplot as plt
from icecream import ic
#/ FRANKA EMIKA 的反解

pi = np.pi
d_bs, d_se, d_ew, d_wt = 333, 316, 384, 107
# * 机器人参数的输入
offset = 88
bias = 82.5  #- 三轴偏移

#* 为了计算角度范围，将某几个矩阵设置为全局变量
A_s, B_s, C_s = [], [], []
A_w, B_w, C_w = [], [], []

# * dh参数中的alpha参数
global alpha
alpha = np.array([-1, 1, 1, -1, 1, 1, 0]) * pi / 2

# * 向量对应斜对称矩阵
def skew_vector(v):
    matrix = np.array([[0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                        [-v[1], v[0], 0]])
    return matrix



def solution_theta_4(kesai, choice):
    k = bias
    b = d_se
    d = d_ew

    a2 = 4 * k**2 + (b-d)**2 - kesai**2
    a1 = 4 * k * (b+d)
    a0 = (d+b)**2 - kesai**2

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

def rot_z(theta):
    '''输入为弧度'''
    result = np.eye(4)
    result[0, 0] = cos(theta)
    result[0, 1] = -sin(theta)
    result[1, 0] = sin(theta)
    result[1, 1] = cos(theta)
    return result

def rot_y(theta):
    '''输入为弧度'''
    result = np.eye(4)
    result[0, 0] = cos(theta)
    result[0, 2] = -sin(theta)
    result[2, 0] = sin(theta)
    result[2, 2] = cos(theta)
    return result

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

def inverse_with_phi(x, r, phi, p_bs, p_se, p_ew, p_wt, lamda):
    cos_7 = np.cos(np.radians(lamda))
    sin_7 = np.sin(np.radians(lamda))
    len_0_bs = np.array([0, 0, p_bs])
    len_3_se_0 = np.array([0, p_se, 0])
    len_4_ew_0 = np.array([0, 0, p_ew])
    len_7_wt = np.array([0, 0, p_wt])
    x_0_sw = x - len_0_bs - r @ len_7_wt

    kesai = np.linalg.norm(x_0_sw) 
    theta_4 = solution_theta_4(kesai, -1)  #/此处有分支
    # theta_4 = solution_theta_4(kesai, -1) + pi  #/此处又有分支
    omega = - theta_4 / 2
    delta_d = np.tan(omega) * bias 
    len_3_se = len_3_se_0 + np.array([0, delta_d, 0])
    len_4_ew = len_4_ew_0 + np.array([0,  0, delta_d])

    # ic(np.degrees(theta_4))
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

    #! 计算theta5,6,7, 由于角7给定，故角6的象限可以去确定，需要分类
    if lamda != 90 and lamda != 270:
        judgement = r_47[2, 0] * cos_7
    else:
        judgement = - r_47[2, 1] * sin_7
    if judgement >= 0:  #- > 还是 >= 待定
        theta_5 = np.arctan2(r_47[1, 2] ,r_47[0, 2])
        theta_6 = np.arccos(-r_47[2, 2])
        theta_7 = np.arctan2(-r_47[2, 1] ,r_47[2, 0])
    else:
        theta_5 = np.arctan2(-r_47[1, 2] ,-r_47[0, 2])
        theta_6 = -np.arccos(-r_47[2, 2])
        theta_7 = np.arctan2(r_47[2, 1] ,-r_47[2, 0])
    # ic(np.tan(theta_7))
    return np.array([theta_1, theta_2, theta_3, theta_4, theta_5, theta_6, theta_7])

def inverse_paramtric_matrix(x, r, p_bs, p_se, p_ew, p_wt):
    len_0_bs = np.array([0, 0, p_bs])
    len_3_se_0 = np.array([0, p_se, 0])
    len_4_ew_0 = np.array([0, 0, p_ew])
    len_7_wt = np.array([0, 0, p_wt])
    x_0_sw = x - len_0_bs - r @ len_7_wt
    kesai = np.linalg.norm(x_0_sw) 
    theta_4 = solution_theta_4(kesai, -1)  #/此处有分支
    # theta_4 = solution_theta_4(kesai, -1) + pi  #/此处有分支
    omega = - theta_4 / 2
    delta_d = np.tan(omega) * bias 
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

    # 腕关节角计算
    # 相应矩阵
    A_w = rotation_axis(theta_4, 4).T @ A_s.T @ r
    B_w = rotation_axis(theta_4, 4).T @ B_s.T @ r
    C_w = rotation_axis(theta_4, 4).T @ C_s.T @ r
    return (A_w, B_w, C_w)


def phi_calculation_sqrt(lamda, M_w, choice):

    upper = (M_w[0][2, 1], M_w[1][2, 1], M_w[2][2, 1])
    lower = (M_w[0][2, 0], M_w[1][2, 0], M_w[2][2, 0])
    eta = np.radians(lamda)
    if lamda != 90 or lamda != 270:   #- 实际应该用同余判断
        coefficient = np.array(upper) + np.tan(eta) * np.array(lower)
    else:
        coefficient = np.array(upper)
    a = coefficient[0]**2 + coefficient[1]**2
    b = 2 * coefficient[0] * coefficient[2]
    c = coefficient[2]**2 - coefficient[1]**2
    square_delta = b**2 - 4*a*c
    if square_delta >= 0:
        sin_phi = (-b + choice * np.sqrt(b**2 - 4*a*c)) / (2*a) #/ 正负根都可以  
        cos_phi = np.sqrt(1 - sin_phi**2)
        return np.arctan2(sin_phi, cos_phi)
    else:
        return None


def phi_calculation(lamda, M_w, choice):
    #- 采用此方法
    #/ 两种算法都可以,前一种容易避开奇异点，后一种更容易让结果连续
    if lamda != 90 or lamda != 270:   #- 实际应该用同余判断
        upper = (M_w[0][2, 1], M_w[1][2, 1], M_w[2][2, 1])
        lower = (M_w[0][2, 0], M_w[1][2, 0], M_w[2][2, 0])
        eta = np.radians(lamda)
        if lamda != 90 or lamda != 270:   #- 实际应该用同余判断
            coefficient = np.array(upper) + np.tan(eta) * np.array(lower)
        else:
            coefficient = np.array(upper)
        a = coefficient[0]
        b = coefficient[1]
        c = coefficient[2]
        d = np.sqrt(a**2 + b**2)
        gamma = np.arctan2(b, a)
        zeta = -c/d
        if np.abs(d) >= np.abs(c):
            if choice == 1:
                phi = np.arcsin(zeta) - gamma
            else:
                phi = np.pi - np.arcsin(zeta) - gamma  
            return phi
        else:
            return None
    else:
        return None

def model_trans(r, beta, lamda):
    eta = np.radians(lamda)
    return r @ rot_z(-eta)[:3, :3] @ rot_y(-beta)[:3, :3] @ rot_z(eta)[:3, :3]

def inverse_with_lamda(x, r, lamda, choice):
    beta = np.arctan(offset / d_wt) 
    sevenlink = np.sqrt(offset ** 2 + d_wt ** 2)
    r_srs = model_trans(r, beta, lamda)
    M_w = inverse_paramtric_matrix(x, r_srs, d_bs, d_se, d_ew, sevenlink)
    if M_w is not None:
        phi = phi_calculation(lamda, M_w, choice)
        if phi is not None:
            result = inverse_with_phi(x, r_srs, phi, d_bs, d_se, d_ew, sevenlink, lamda)
            result[5] -= beta
            return result
        else:
            # print('Not solvable:{0}'.format(lamda))
            return None
    else:
        # print('Not solvable:{0}'.format(lamda))
        return None

def inverse_for_phi(x, r, lamda, choice):
    beta = np.arctan(offset / d_wt) 
    sevenlink = np.sqrt(offset ** 2 + d_wt ** 2)
    r_srs = model_trans(r, beta, lamda)
    M_w = inverse_paramtric_matrix(x, r_srs, d_bs, d_se, d_ew, sevenlink)
    if M_w is not None:
        if lamda > 90 and lamda < 270:
            phi = phi_calculation(lamda, M_w, choice)
        elif lamda != 90 and lamda != 270:
            phi = phi_calculation(lamda, M_w, -choice)
        else:
            phi = phi_calculation(lamda, M_w, choice)
        return phi
    else:
        return None

def inverse_kinematics(x, r, lamda, branch):
    #- 解的拼接
    result_pos = inverse_with_lamda(x, r, lamda, 1)
    result_neg = inverse_with_lamda(x, r, lamda, -1)
    if result_pos is not None:
        result_pos = normalize_theta_output(result_pos)
        result_neg = normalize_theta_output(result_neg)

    if lamda > 90 and lamda < 270:
        return result_neg if branch == 1 else result_pos
    elif lamda != 90 and lamda != 270:
        return result_pos if branch == 1 else result_neg
    elif lamda == 90:
        return result_pos if branch == 1 else result_neg
    elif lamda == 270:
        return result_neg if branch == 1 else result_pos

def mdh_mat(theta, d, a, alpha):
    '''分别是dh参数的a d theta alpha'''
    matrix = np.array([
        [cos(theta), -sin(theta),  0, a],
        [cos(alpha) * sin(theta),  cos(alpha) * cos(theta), -sin(alpha), -d * sin(alpha)],
        [sin(alpha) * sin(theta),  sin(alpha) * cos(theta),  cos(alpha),  d * cos(alpha)],
        [0, 0, 0, 1]
        ])
    return matrix

def fk_kuka(ang):
    Joint1 = mdh_mat(ang[0], d_bs, 0, 0)
    Joint2 = mdh_mat(ang[1], 0, 0, -pi/2)
    Joint3 = mdh_mat(ang[2], d_se, 0, pi/2)
    Joint4 = mdh_mat(ang[3], 0, bias, pi/2)
    Joint5 = mdh_mat(ang[4], d_ew, -bias, -pi/2)
    Joint6 = mdh_mat(ang[5], 0, 0, pi/2) 
    Joint7 = mdh_mat(ang[6], 0, offset, pi/2)
    Flange = mdh_mat(0, d_wt, 0, 0)
    Posture = Joint1 @ Joint2 @ Joint3 @ Joint4 @ Joint5 @ Joint6 @ Joint7 
    return Posture @ Flange

def show_error(result, phi, r, x):
    error_rotation = result[:3, :3] @ np.linalg.inv(r) - np.eye(3)
    error_translation = result[:3, 3] - x
    print('degree {} is solvable'.format(phi))
    print('error of r: ', np.linalg.norm(error_rotation))
    print('error of x: ', np.linalg.norm(error_translation))

def normalize_theta_output(theta):
        theta = np.degrees(theta)
        #/ 限制角度的范围在upper和lower之间
        # theta = theta % 360
        for i in range(7):
            if i != 3 and i != 5:
                while True:
                    if (-181 <= theta[i]) and (181 >= theta[i]):
                        break
                    elif -181 > theta[i]:
                        theta[i] += 360
                    else:
                        theta[i] -= 360
            elif i == 3:
                while True:
                    if (-271 <= theta[i]) and (91 >= theta[i]):
                        break
                    elif -271 > theta[i]:
                        theta[i] += 360
                    else:
                        theta[i] -= 360
            elif i == 5:
                while True:
                    if (-74 <= theta[i]) and (288 >= theta[i]):
                        break
                    elif -74 > theta[i]:
                        theta[i] += 360
                    else:
                        theta[i] -= 360
        return np.radians(theta) 

if __name__ == '__main__':
    # 测试论文结果使用的位姿
    # x_07_d = np.array([500, 180, 700])
    # r_07_d = np.array([[0.067, 0.933, 0.354],
    #                 [0.933, 0.067, -0.354],
    #                 [-0.354, 0.354, -0.866]])
    # x = x_07_d
    # r = r_07_d
    tst_theta = np.radians([-27, 50, 57, -98, 69, 131, -79])
    reference = fk_kuka(tst_theta)
    x = reference[:3, 3]
    r = reference[:3, :3]

    # import time
    # start_time = time.time()
    # for i in range(1000):
    # angle = inverse_with_lamda(x, r, -79, -1)  #TODO 查看公式，双输入反正切其实依赖角6，角2的sin
    #- 0.5ms一次计算
    # print(time.time() - start_time)
    # ic(np.degrees(angle))
    # ic(fk_kuka(angle))
    # ic(x)
    # ic(r)   #TODO 多值引起的角度不匹配与解丢失

    phi_set = []
    joints = []
    joints2 = []
    real_phi = []
    real_phi2 = []

    for phi in list(np.linspace(0, 360, 100)):
        if phi == 90 or phi == 270:
            continue
        # angle = inverse_with_lamda(x, r, phi, -1)
        # angle2 = inverse_with_lamda(x, r, phi, 1)
        angle = inverse_kinematics(x, r, phi, -1)
        angle2 = inverse_kinematics(x, r, phi, 1)

        if angle is not None:
            phi_set.append(phi)
            joints.append(angle)
            joints2.append(angle2)
            real_phi.append(inverse_for_phi(x, r, phi, 1))
            real_phi2.append(inverse_for_phi(x, r, phi, -1))
            result = fk_kuka(angle)
            # show_error(result, phi, r, x)
    joints = np.array(joints)
    joints2 = np.array(joints2)
    real_phi = np.array(real_phi)
    real_phi2 = np.array(real_phi2)
    # reformulate angles
    for i in range(7):
        for j in range(len(phi_set) - 1):
            diff = joints[j+1, i] - joints[j, i]
            if diff > 1.95 * pi:
                joints[j+1, i] -= 2 * pi
            elif diff < -1.95 * pi:
                joints[j+1, i] += 2 * pi

    for i in range(7):
        for j in range(len(phi_set) - 1):
            diff = joints2[j+1, i] - joints2[j, i]
            if diff > 1.95 * pi:
                joints2[j+1, i] -= 2 * pi
            elif diff < -1.95 * pi:
                joints2[j+1, i] += 2 * pi

    np.save('route/impedence.npy', joints)
    # for j in range(len(phi_set) - 1):
    #     diff = real_phi[j+1] - real_phi[j]
    #     if diff > 1.95 * pi:
    #         real_phi[j+1] -= 2 * pi
    #     elif diff < -1.95 * pi:
    #         real_phi[j+1] += 2 * pi

    # for j in range(len(phi_set) - 1):
    #     diff = real_phi2[j+1] - real_phi2[j]
    #     if diff > 1.95 * pi:
    #         real_phi2[j+1] -= 2 * pi
    #     elif diff < -1.95 * pi:
    #         real_phi2[j+1] += 2 * pi

    for i in range(7):
        # plt.scatter(phi_set, np.degrees(joints[:, i]), label='joint {0}'.format(i+1), s=2)
        # plt.scatter(phi_set, np.degrees(joints2[:, i]), label='joint2 {0}'.format(i+1), s=2)      
        plt.plot(phi_set, np.degrees(joints[:, i]), label='joint {0}'.format(i+1))
        # plt.plot(phi_set, np.degrees(joints2[:, i]), label='joint2 {0}'.format(i+1))
    plt.legend()
    plt.show()

    # # plt.scatter(phi_set, np.degrees(real_phi), label='phi', s=2)
    # # plt.scatter(phi_set, np.degrees(real_phi2), label='phi2', s=2)
    # # plt.legend()
    # # plt.show()
