import numpy as np
import casadi as ca

##################################### 几何参数 ###########################
d_bs, d_se, d_ew, d_wt = 0.333, 0.316, 0.384, 0.107
offset = 0.088
bias = 0.0825

A_s, B_s, C_s = [], [], []
A_w, B_w, C_w = [], [], []

alpha = np.array([-1, 1, 1, -1, 1, 1, 0]) * np.pi / 2


##################################### 通用辅助函数 ##########################
def rot_z(theta):
    result = ca.SX_eye(3)
    result[0, 0] = ca.cos(theta)
    result[0, 1] = -ca.sin(theta)
    result[1, 0] = ca.sin(theta)
    result[1, 1] = ca.cos(theta)
    return result

def rot_y(theta):
    result = ca.SX_eye(3)
    result[0, 0] = ca.cos(theta)
    result[0, 2] = -ca.sin(theta)
    result[2, 0] = ca.sin(theta)
    result[2, 2] = ca.cos(theta)
    return result

def skew(v):
    matrix = ca.skew(v)
    return matrix


#- 基于法兰位置而非工具位置
def vector_rot(base, to):
    cs = ca.dot(base, to)
    axis = ca.cross(base, to)
    sn = ca.norm_2(axis)

    # if (sn < 0.001):
    #     return np.eye(3)
    axis = axis / sn

    m1 = skew(axis)
    m2 = ca.dot(m1, m1)
    rot = ca.SX_eye(3) + sn * m1 + (1 - cs) * m2
    return rot


################################### 辅助函数 #################################
def joint_rotation(q, index):
    cp = ca.cos(alpha[index - 1])
    sp = ca.sin(alpha[index - 1])
    cq = ca.cos(q)
    sq = ca.sin(q)
    
    rotation =  ca.SX.zeros(3, 3)
    rotation[0, 0] = cq
    rotation[0, 1] = -sq * cp
    rotation[0, 2] = sq * sp
    
    rotation[1, 0] = sq
    rotation[1, 1] = cq * cp
    rotation[1, 2] = -cq * sp
    
    rotation[2, 1] = sp
    rotation[2, 2] = cp
    return rotation


################################### 反解函数 ##################################

#- q2为正
#- zeta = 当前q7
def ik(target, zeta):

    x = target[:3, 3]
    r_franka = target[:3, :3]

    #* 消除q7带来的相对srs构型的偏移
    beta = ca.arctan(offset / d_wt)

    length_link7 = ca.sqrt(offset**2 + d_wt**2)
    r = r_franka @ rot_z(-zeta) @ rot_y(-beta) @ rot_z(zeta)

    #- 位置向量
    vec_0_bs = np.array([0, 0, d_bs])
    vec_3_se_franka = np.array([0, d_se, 0])
    vec_4_ew_franka = np.array([0, 0, d_ew])
    vec_7_wt = np.array([0, 0, length_link7])

    #* 计算q4
    vec_0_sw = x - vec_0_bs - r @ vec_7_wt
    d_sw = ca.norm_2(vec_0_sw)

    a2 = 4 * bias**2 + (d_se - d_ew)**2 - d_sw**2
    a1 = 4 * bias * (d_se + d_ew)
    a0 = (d_se + d_ew)**2 - d_sw**2

    #. 判别式小于0则无解
    tan_half_q4 = (a1 + ca.sqrt(a1**2 - 4 * a0 * a2)) / (2 * a2)
    q4 = ca.arctan(tan_half_q4) * 2

    #* 从一族解中选择对应于q7的解，为此需要计算参数phi
    #- 构造矩阵M_w, 使用M_w和输入q7计算phi
    #- q4已知，得到srs构型，但是连杆长度有所变化
    delta_d = ca.tan(-q4 / 2) * bias
    d_vec3 = ca.SX(3, 1)
    d_vec3[1] = delta_d
    vec_3_se = vec_3_se_franka + d_vec3
    d_vec4 = ca.SX(3, 1)
    d_vec4[2] = delta_d
    vec_4_ew = vec_4_ew_franka + d_vec4

    u_0_sw = vec_0_sw / ca.norm_2(vec_0_sw)
    x_aux = joint_rotation(0, 3) @ (vec_3_se + joint_rotation(q4, 4) @ vec_4_ew)
    y_aux = vec_0_sw

    dq1 = ca.arctan2(y_aux[1], y_aux[0])    
    dq2 = ca.arctan2(x_aux[2], x_aux[0])
    ampl1 = ca.sqrt(x_aux[0]**2 + x_aux[2]**2)
    ampl2 = ca.sqrt(y_aux[0]**2 + y_aux[1]**2)
    q1_ref = ca.arcsin(-x_aux[1] / ampl2) + dq1
    q2_ref = ca.arcsin(-y_aux[2] / ampl1) + dq2
    
    r_03_ref = joint_rotation(q1_ref, 1) @ joint_rotation(q2_ref, 2) @ joint_rotation(0, 3)
    skew_sw = skew(u_0_sw)
    # shoulder
    A_s = skew_sw @ r_03_ref
    B_s = - skew_sw @ A_s
    C_s = u_0_sw.reshape((3, 1)) @ u_0_sw.reshape((1, 3)) @ r_03_ref
    # wrist
    R4 = joint_rotation(q4, 4)
    A_w = R4.T @ A_s.T @ r
    B_w = R4.T @ B_s.T @ r
    C_w = R4.T @ C_s.T @ r
    
    
    #. tan(q7) 为无穷则需要调整
    coeff = ca.SX(3, 1)
    coeff[0] = A_w[2, 1] + ca.tan(zeta) * A_w[2, 0]
    coeff[1] = B_w[2, 1] + ca.tan(zeta) * B_w[2, 0]
    coeff[2] = C_w[2, 1] + ca.tan(zeta) * C_w[2, 0]
    
    sin_aux = - coeff[2] / ca.sqrt(coeff[0]**2 + coeff[1]**2)
    #. sin_aux 幅值大于1则无解    
    phi = np.pi - ca.arctan2(coeff[1], coeff[0]) - ca.arcsin(sin_aux)
    
    #* 使用phi求其他六个角
    r_03 = A_s * ca.sin(phi) + B_s * ca.cos(phi) + C_s
    r_47 = A_w * ca.sin(phi) + B_w * ca.cos(phi) + C_w
    
    #* 计算q1~3
    #. 角2的正负可以导致两组解
    q1 = ca.arctan2(r_03[1, 1], r_03[0, 1])
    q2 = ca.arccos(r_03[2, 1])
    q3 = ca.arctan2(-r_03[2, 2], -r_03[2, 0])
    
    criteria = r_47[2, 0] * ca.cos(zeta)
    q5 = ca.arctan2(r_47[1, 2] ,r_47[0, 2])
    q6 = ca.arccos(-r_47[2, 2])

    q_aux = ca.SX(9, 1)
    q_aux_list = [q1, q2, q3, q4, q5, q6, zeta, beta, criteria]
    for i in range(9):
        q_aux[i] = q_aux_list[i]

    return q_aux


def generate_ik():
    zeta = ca.SX.sym('zeta', 1)
    target = ca.SX.sym('target', 4, 4)
    q_aux = ik(target, zeta)
    ik_aux = ca.Function('ik', [target, zeta], [q_aux])
    return ik_aux

f = generate_ik()

def ik_ca(target, zeta):
    q_aux = np.array(f(target, zeta))
    q = q_aux[:-2]
    beta = q_aux[-2]
    criteria = q_aux[-1]
    
    #* 加上偏移量
    if criteria < 0:
        q[4] += np.pi
        q[5] = -q[5]
    q[5] -= beta
    return q.reshape(7)

def ik_ca_neg(target, zeta):
    q_aux = np.array(f(target, zeta))
    q = q_aux[:-2]
    beta = q_aux[-2]
    criteria = q_aux[-1]
    
    #* 加上偏移量
    if criteria < 0:
        q[4] += np.pi
        q[5] = -q[5]
    q[5] -= beta

    q[0] = np.arctan2(-np.sin(q[0]), -np.cos(q[0]))
    q[1] = - q[1]
    q[2] = np.arctan2(-np.sin(q[2]), -np.cos(q[2]))

    return q.reshape(7)

def limit_joints(theta):
    theta = np.degrees(theta)

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
    pass