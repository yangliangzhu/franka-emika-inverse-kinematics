import casadi as ca
import numpy as np
from numpy import pi
## another kind of franka emika model

class Panda():
    '''
    franka emika dh parameters
    '''
    dh_params = np.array([[      0,  0.333,     0,  0],
                          [      0,      0, -pi/2,  0],
                          [      0,  0.316,  pi/2,  0],
                          [ 0.0825,      0,  pi/2,  0],
                          [-0.0825,  0.384, -pi/2,  0],
                          [      0,      0,  pi/2,  0],
                          [  0.088,      0,  pi/2,  0],
                          [      0,  0.107,     0,  0],
                          [      0, 0.1034,     0,  0]])
    
    def __init__(self, num_dof=7, jactype=0):
        
        self.num_dof = num_dof
        if jactype == 0:
            self.jacobian, self.jacobian_pinv = self.jacobian_generator()
        else:
            self.jacobian, self.jacobian_pinv = self.jacobian_generator2()
        self.jacobian_flange, self.jacobian_flange_pinv = self.jacobian_flange_generator()
        self.forward = self.fk_generator()
        self.forward_flange = self.fk_flange_generator()
        self.diff_homomatrix_nonsin = self.generate_diff_homomatrix()
        self.diff_homomatrix_sin = self.generate_diff_homomatrix_sin()

        self.__upper_bounds = np.radians([165, 100, 165, -5, 165, 214, 165])
        self.__lower_bounds = np.radians([-165, -100, -165, -175, -165, 0, -165])
        
    @property
    def upper_bounds(self):
        return self.__upper_bounds
    
    @property
    def lower_bounds(self):
        return self.__lower_bounds
        
    def forward_kinematics(self, joints):
        '''
        Unit: rad
        '''
        forward_kinematics = []
        last_T = np.identity(4)
        for i in range(self.dh_params.shape[0]):
            a, d, alpha, _ = self.dh_params[i]
            if i < 7:
                theta = joints[i]
            else:
                theta = 0
            rotX = self.rot_x(alpha)
            transX = self.trans_x(a)
            transZ = self.trans_z(d)
            rotZ = self.rot_z(theta)

            T = rotX @ transX @ transZ @ rotZ
            T = last_T @T

            last_T = T
            forward_kinematics.append(T)
        return forward_kinematics
    
    def fk(self, joints):
        '''
        return the position of the end-effector.
        Unit: rad
        '''
        dh_fk = self.forward_kinematics(joints)
        H = dh_fk[-1]
        Tool = self.rot_z(-pi/4)
        return H @ Tool
    
    def fk_flange(self, joints):
        '''
        return the position of the flange.
        Unit: rad
        '''
        last_T = np.identity(4)
        for i in range(self.dh_params.shape[0] - 1):
            a, d, alpha, _ = self.dh_params[i]
            if i < 7:
                theta = joints[i]
            else:
                theta = 0
            rotX = self.rot_x(alpha)
            transX = self.trans_x(a)
            transZ = self.trans_z(d)
            rotZ = self.rot_z(theta)

            T = rotX @ transX @ transZ @ rotZ
            T = last_T @ T

            last_T = T
        return last_T
    
    def fk_generator(self):
        q = ca.SX.sym('q', 7)
        rhsx = self.fk(q)
        forward = ca.Function('forward', [q], [rhsx])
        return forward
    
    def fk_flange_generator(self):
        q = ca.SX.sym('q', 7)
        rhsx = self.fk_flange(q)
        forward_f = ca.Function('forward_f', [q], [rhsx])
        return forward_f
    
    def jacobian_generator(self):
        #using velocity passing
        q = ca.SX.sym('q', 7)
        jacobian = ca.SX.zeros((6, self.num_dof))
        dh_fk = self.forward_kinematics(q)
        
        for i in range(self.num_dof):
            offset = dh_fk[-1][:-1, -1] - dh_fk[i][:-1, -1]
            angular_v = dh_fk[i][:-1, :-1] @ [0, 0, 1]
            linear_v = ca.cross(angular_v, offset)
            jacobian[:, i] = ca.vertcat(linear_v, angular_v)
            
        f = ca.Function('f', [q], [jacobian])
        
        jacobian_p = ca.pinv(jacobian)
        fp = ca.Function('f', [q], [jacobian_p])
        
        return f, fp
    
    def jacobian_generator2(self):
        #using lie algebra
        q = ca.SX.sym('q', 7)
        rhsx = self.fk(q)[:3, 3]
        jacx = ca.jacobian(rhsx, q)

        row1 = self.fk(q)[0, :3]
        row2 = self.fk(q)[1, :3]
        row3 = self.fk(q)[2, :3]

        omegax = row2 @ ca.jacobian(row3, q)
        omegay = row3 @ ca.jacobian(row1, q)
        omegaz = row1 @ ca.jacobian(row2, q)

        rhs = ca.vertcat(jacx, omegax, omegay, omegaz)
        f = ca.Function('f', [q], [rhs])

        rhsp = ca.pinv(rhs)
        fp = ca.Function('f', [q], [rhsp])

        return f, fp
    
    def jacobian_flange_generator(self):
        #using lie algebra
        q = ca.SX.sym('q', 7)
        rhsx = self.fk_flange(q)[:3, 3]
        jacx = ca.jacobian(rhsx, q)

        row1 = self.fk(q)[0, :3]
        row2 = self.fk(q)[1, :3]
        row3 = self.fk(q)[2, :3]

        omegax = row2 @ ca.jacobian(row3, q)
        omegay = row3 @ ca.jacobian(row1, q)
        omegaz = row1 @ ca.jacobian(row2, q)

        rhs = ca.vertcat(jacx, omegax, omegay, omegaz)
        f = ca.Function('f', [q], [rhs])

        rhsp = ca.pinv(rhs)
        fp = ca.Function('f', [q], [rhsp])

        return f, fp
    
    def ik(self, angle, tar_T):
        gain = 0.8
        pre_T = self.forward(angle)
        t = 0
        diff = self.diff_homomatrix(
                pre_T, tar_T).reshape((6, 1))
        dis1 = ca.norm_2(diff[:3])
        dis2 = ca.norm_2(diff[3:])
        
        while (dis1 > 0.0002 or dis2 > 0.02) and (t < 50):
            
            dt = self.jacobian_pinv(angle) @ diff

            angle += dt.reshape((7, 1)) * gain
            pre_T = self.forward(angle)
            t = t + 1
            diff = self.diff_homomatrix(
                    pre_T, tar_T).reshape((6, 1))
            dis1 = ca.norm_2(diff[:3])
            dis2 = ca.norm_2(diff[3:])
        if t == 50:
            raise (TimeoutError('the result diverges'))
        # return self.normalize_theta(np.array(angle)).reshape(7), t
        return self.normalize_theta(np.array(angle)).reshape(7)

    
    def diff_homomatrix(self, A, B):
        if np.allclose(A[:3, :3], B[:3, :3], atol=0.01):
            return self.diff_homomatrix_sin(A, B)
        else:
            return self.diff_homomatrix_nonsin(A, B)

    def generate_diff_homomatrix(self):
        A = ca.SX.sym('A', 4, 4)
        B = ca.SX.sym('B', 4, 4)
        rhs = self.diff_homomatrix_nonsingular(A, B)
        diff_matrix = ca.Function('diff_matrix', [A, B], [rhs])
        return diff_matrix
    
    def generate_diff_homomatrix_sin(self):
        A = ca.SX.sym('A', 4, 4)
        B = ca.SX.sym('B', 4, 4)
        rhs = self.diff_homomatrix_singular(A, B)
        diff_matrix_sin = ca.Function('diff_matrix_sin', [A, B], [rhs])
        return diff_matrix_sin
    
    
    @staticmethod
    def rot_x(alpha):
        matrix =  ca.SX_eye(4)
        matrix[1, 1] = ca.cos(alpha)
        matrix[1, 2] = - ca.sin(alpha)
        matrix[2, 1] = ca.sin(alpha)
        matrix[2, 2] = ca.cos(alpha)
        return matrix
    
    @staticmethod
    def rot_z(theta):
        matrix =  ca.SX_eye(4)
        matrix[0, 0] = ca.cos(theta)
        matrix[0, 1] = - ca.sin(theta)
        matrix[1, 0] = ca.sin(theta)
        matrix[1, 1] = ca.cos(theta)
        return matrix
    
    @staticmethod
    def trans_x(a):
        matrix = ca.SX_eye(4)
        matrix[0, 3] = a
        return matrix
    
    @staticmethod
    def trans_z(d):
        matrix = ca.SX_eye(4)
        matrix[2, 3] = d
        return matrix
    
    @staticmethod
    #. 需要调整为franka的角度范围
    def normalize_theta(theta):
        theta = np.degrees(theta)
        theta = theta % 360
        for i in range(7):
            if (-180 <= theta[i]) and (180 >= theta[i]):
                continue
            elif -180 > theta[i]:
                theta[i] += 360
                continue
            else:
                theta[i] -= 360
        return np.radians(theta)

    @staticmethod
    def diff_homomatrix_nonsingular(A, B):
        dx = (B - A)[0:3, 3]
        dr = B[0:3, 0:3] @ ca.inv(A[0:3, 0:3])

        phi = ca.acos((dr[0, 0] + dr[1, 1] + dr[2, 2] - 1) / 2)
        omega = phi / (2 * np.sin(phi)) * (dr - dr.T)
        w1 = omega[2, 1]
        w2 = omega[0, 2]
        w3 = omega[1, 0]
        dt = ca.horzcat(dx[0], dx[1], dx[2], w1, w2, w3)
        return dt

    @staticmethod
    def diff_homomatrix_singular(A, B):
        dx = (B - A)[0:3, 3]   
        dt = ca.horzcat(dx[0], dx[1], dx[2], 0, 0, 0)
        return dt

if __name__ == '__main__':
    '''
    用例
    '''
    q = ca.SX.sym('q', 7)

    rob = Panda()

    np.set_printoptions(precision=3, suppress=True)
    tar_T = rob.forward([1, 2, 2, 1, 0, 0, 0])
    pre_T = rob.forward([1, 1, 2, 1, 1, 3, 1])
    print(np.array(tar_T))
    print(np.array(rob.jacobian([1, 1, 2, 1, 1, 3, 1])))
    
    import time
    start = time.time()
    for i in range(1):
        res = rob.ik([10, 10, 20, 10, 10, 30, 10], tar_T)
    end = time.time()
    print(end - start)
    print(np.array(res))

        
        