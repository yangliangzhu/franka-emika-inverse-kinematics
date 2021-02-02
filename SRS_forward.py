import numpy as np
from numpy import cos, sin, pi
# import application.transform as tf
# import matplotlib.pyplot as plt 
# from rotation_visualization import visualize_frame


def mdh_mat(theta, d, a, alpha):
    '''分别是dh参数的a d theta alpha'''
    matrix = np.array([
        [cos(theta), -sin(theta),  0, a],
        [cos(alpha) * sin(theta),  cos(alpha) * cos(theta), -sin(alpha), -d * sin(alpha)],
        [sin(alpha) * sin(theta),  sin(alpha) * cos(theta),  cos(alpha),  d * cos(alpha)],
        [0, 0, 0, 1]
        ])
    return matrix

def fk_prototype(ang):
    Joint1 = mdh_mat(ang[0], 333, 0, 0)
    Joint2 = mdh_mat(ang[1], 0, 0, -pi/2)
    Joint3 = mdh_mat(ang[2], 316, 0, pi/2)
    Joint4 = mdh_mat(ang[3], 0, 0, pi/2)
    Joint5 = mdh_mat(ang[4], 384, 0, -pi/2)
    Joint6 = mdh_mat(ang[5], 0, 0, pi/2) 
    Joint7 = mdh_mat(ang[6], 107, 0, pi/2)
    Flange = mdh_mat(0, 0, 0, 0)
    Posture = Joint1 @ Joint2 @ Joint3 @ Joint4 @ Joint5 @ Joint6 @ Joint7 @ Flange
    return Posture

def fk_map(ang):
    alpha = - ang[3] / 2
    delta_d = np.tan(alpha) * 82.5                                                                
    seven_link = np.sqrt(88 ** 2 + 107 ** 2)
    beta = np.arctan(88 / 107)                          
                                                                                                                        
    Joint1 = mdh_mat(ang[0], 333, 0, 0)
    Joint2 = mdh_mat(ang[1], 0, 0, -pi/2)
    Joint3 = mdh_mat(ang[2], 316 + delta_d, 0, pi/2)                                     
    Joint4 = mdh_mat(ang[3], 0, 0, pi/2)
    Joint5 = mdh_mat(ang[4], 384 + delta_d, 0, -pi/2)
    Joint6 = mdh_mat(ang[5] + beta, 0, 0, pi/2) 
    Joint7 = mdh_mat(ang[6], seven_link, 0, pi/2)

    Flange = mdh_mat(0, 0, 0, 0) 
    Posture = Joint1 @ Joint2 @ Joint3 @ Joint4 @ Joint5 @ Joint6 @ Joint7 @ Flange
    Posture[:3, :3] = Posture[:3, :3] @ rot_z(-ang[6])[:3, :3] @ rot_y(beta)[:3, :3] @ rot_z(ang[6])[:3, :3]
    return Posture

# def fk_map_kuka(ang):
#     #/ kuka
#     alpha =  ang[3] / 2  #- 视情况取正负
#     delta_d = np.tan(alpha) * 82.5 
#     seven_link = np.sqrt(88 ** 2 + 70 ** 2)
#     beta = np.arctan(88 / 70)

#     Joint1 = mdh_mat(ang[0], 317, 0, 0)
#     Joint2 = mdh_mat(ang[1], 0, 0, -pi/2)
#     Joint3 = mdh_mat(ang[2], 450 + delta_d, 0, pi/2)
#     Joint4 = mdh_mat(ang[3], 0, 0, -pi/2)
#     Joint5 = mdh_mat(ang[4], 480 + delta_d, 0, pi/2)
#     Joint6 = mdh_mat(ang[5] + beta, 0, 0, -pi/2) 
#     Joint7 = mdh_mat(ang[6], seven_link, 0, pi/2)

#     Flange = mdh_mat(0, 0, 0, 0) 
#     Posture = Joint1 @ Joint2 @ Joint3 @ Joint4 @ Joint5 @ Joint6 @ Joint7 @ Flange
#     Posture[:3, :3] = Posture[:3, :3] @ rot_z(-ang[6])[:3, :3] @ rot_y(beta)[:3, :3] @ rot_z(ang[6])[:3, :3]
#     return Posture

def fk_alter_elbow(ang):
    Joint1 = mdh_mat(ang[0], 333, 0, 0)
    Joint2 = mdh_mat(ang[1], 0, 0, -pi/2)
    Joint3 = mdh_mat(ang[2], 316, 0, pi/2)
    Joint4 = mdh_mat(ang[3], 0, 82.5, pi/2)
    Joint5 = mdh_mat(ang[4], 384, -82.5, -pi/2)
    Joint6 = mdh_mat(ang[5], 0, 0, pi/2) 
    Joint7 = mdh_mat(ang[6], 0, 0, pi/2)
    Flange = np.eye(4)
    Posture = Joint1 @ Joint2 @ Joint3 @ Joint4 @ Joint5 @ Joint6 @ Joint7 @ Flange
    return Posture

def fk_alter_wrist(ang):
    Joint1 = mdh_mat(ang[0], 333, 0, 0)
    Joint2 = mdh_mat(ang[1], 0, 0, -pi/2)
    Joint3 = mdh_mat(ang[2], 316, 0, pi/2)
    Joint4 = mdh_mat(ang[3], 0, 0, pi/2)
    Joint5 = mdh_mat(ang[4], 384, 0, -pi/2)
    Joint6 = mdh_mat(ang[5], 0, 0, pi/2) 
    Joint7 = mdh_mat(ang[6], 0, 88, pi/2)
    Flange = mdh_mat(0, 107, 0, 0) 
    Posture = Joint1 @ Joint2 @ Joint3 @ Joint4 @ Joint5 @ Joint6 @ Joint7 @ Flange
    return Posture

def fk_alter_full(ang):
    Joint1 = mdh_mat(ang[0], 333, 0, 0)
    Joint2 = mdh_mat(ang[1], 0, 0, -pi/2)
    Joint3 = mdh_mat(ang[2], 316, 0, pi/2)
    Joint4 = mdh_mat(ang[3], 0, 82.5, pi/2)
    Joint5 = mdh_mat(ang[4], 384, -82.5, -pi/2)
    Joint6 = mdh_mat(ang[5], 0, 0, pi/2) 
    Joint7 = mdh_mat(ang[6], 0, 88, pi/2)
    Flange = mdh_mat(0, 107, 0, 0) 
    Posture = Joint1 @ Joint2 @ Joint3 @ Joint4 @ Joint5 @ Joint6 @ Joint7 @ Flange
    return Posture    

# def fk_alter_full(ang):
#     #/ kuka
#     Joint1 = mdh_mat(ang[0], 317, 0, 0)
#     Joint2 = mdh_mat(ang[1], 0, 0, -pi/2)
#     Joint3 = mdh_mat(ang[2], 450, 0, pi/2)
#     Joint4 = mdh_mat(ang[3], 0, 82.5, -pi/2)
#     Joint5 = mdh_mat(ang[4], 480, -82.5, pi/2)
#     Joint6 = mdh_mat(ang[5], 0, 0, -pi/2) 
#     Joint7 = mdh_mat(ang[6], 0, 88, pi/2)
#     Flange = mdh_mat(0, 70, 0, 0) 
#     Posture = Joint1 @ Joint2 @ Joint3 @ Joint4 @ Joint5 @ Joint6 @ Joint7 @ Flange
#     return Posture    

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


