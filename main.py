import roboticstoolbox as rtb
from math import cos, sin, pi, atan2, sqrt, acos, atan
import numpy as np
import matplotlib.pyplot as plt

class Traj:
    def __init__(self, jointCount : int, gapPoint : int = 50):
        self.t = np.zeros((gapPoint, 1))
        self.q = np.zeros((gapPoint, jointCount))
        self.w = np.zeros((gapPoint, jointCount))
        self.a = np.zeros((gapPoint, jointCount))

class Puma560:
    def __init__(self):
        self.dh_params = [
                {'theta': 0, 'd': 0.6718, 'a': 0, 'alpha': pi/2},
                {'theta': 0, 'd': 0, 'a': 0.4318, 'alpha': 0},
                {'theta': 0, 'd': 0.15005, 'a': 0.0203, 'alpha': -pi/2},
                {'theta': 0, 'd': 0.4318, 'a': 0, 'alpha': pi/2},
                {'theta': 0, 'd': 0, 'a': 0, 'alpha': -pi/2},
                {'theta': 0, 'd': 0, 'a': 0, 'alpha': 0},
            ]
        self.limits = [(-160, 160), (-110, 110), (-135, 135), (-266, 266), (-100, 100), (-266, 266)] # 关节角限制
    
    def dh_matrix(self, theta: float, d: float, a: float, alpha: float) -> np.ndarray:
        """Puma560六轴机械臂的DH变换矩阵

        Args:
            theta (float): 
            d (float): 
            a (float): 
            alpha (float): 

        Returns:
            np.ndarray: 变换矩阵
        """
        return np.array([
            [cos(theta), -sin(theta)*cos(alpha), sin(theta)*sin(alpha), a*cos(theta)],
            [sin(theta), cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta)],
            [0, sin(alpha), cos(alpha), d],
            [0, 0, 0, 1]
        ])
    
    def forward(self, q: np.ndarray) -> np.ndarray:
        """Puma560六轴机械臂根据DH参数表和关节角度计算正运动学

        Args:
            q (ndarray): 6个关节角度, 要求为弧度

        Returns:
            ndaarray: 末端位姿矩阵
        """
        T_total = np.eye(4)
        for i, params in enumerate(self.dh_params):
            T_i = self.dh_matrix(q[i], params['d'], params['a'], params['alpha'])
            T_total = np.dot(T_total, T_i)
        return T_total

    def limit(self, rad:float, LimitsIndex: int, e=1e-3):
        """判断rad是否超过关节角限制

        Args:
            rad (float): 关节角弧度
            LimitsIndex (int): 索引即第LimitsIndex个关节角
            e (float, optional): 误差范围. Defaults to 1e-3.

        Returns:
            bool: 超过限制返回False，否则返回True
        """
        deg = np.rad2deg(rad)
        Limits = self.limits
        if (deg - Limits[LimitsIndex][0]) < -e or (deg - Limits[LimitsIndex][1]) > e:
            return False
        return True
    
    def inverse(self, T: np.ndarray) -> np.ndarray:
        """根据位姿求解关节角

        Args:
            T (np.ndarray): 目标位姿
        Raises:
            ValueError: 判断T是否为规定的形状

        Returns:
            np.ndarray: 所有符合条件的解，值为弧度
        """
        if T.shape == (3, ):
            nx, ny, nz = (1, 0, 0)
            ox, oy, oz = (0, 1, 0)
            ax, ay, az = (0, 0, 1)
            px, py, pz = T
        elif T.shape == (4, 4):
            nx, ny, nz = T[:3, 0]
            ox, oy, oz = T[:3, 1]
            ax, ay, az = T[:3, 2]
            px, py, pz = T[:3, 3]
        else:
            raise ValueError("T shape must be (3, ) or (4, 4)")
        d1, d3, d4 = self.dh_params[0]['d'], self.dh_params[2]['d'], self.dh_params[3]['d']
        a2, a3 = self.dh_params[1]['a'], self.dh_params[2]['a']
        P = pz - d1
        K = px**2 + py**2 + P**2 - a2**2 - a3**2 - d4**2 - d3**2
        dh_params = self.dh_params
        
        thetaList: list[tuple] = []
        
        ################ 求解 theta1 ################
        theta1_0 = atan2(py, px) - atan2(-d3, sqrt(px**2 + py**2 - dh_params[2]['d']**2))
        theta1_1 = atan2(py, px) - atan2(-d3, -sqrt(px**2 + py**2 - dh_params[2]['d']**2))
        thetaList.append((theta1_0,))
        thetaList.append((theta1_1,))
        
        ################ 求解 theta2 ################
        for index in range(len(thetaList.copy())):
            theta3_0 = atan2(a3, d4) - atan2(K, sqrt((2*a2*d4)**2 + (2*a2*a3)**2 - K**2))
            theta3_1 = atan2(a3, d4) - atan2(K, -sqrt((2*a2*d4)**2 + (2*a2*a3)**2 - K**2))
            theta1, = thetaList.pop(0)
            thetaList.append((theta1, theta3_0))
            thetaList.append((theta1, theta3_1))
            
        ################ 求解 theta3 ################
        for index in range(len(thetaList.copy())):
            theta1, theta3 = thetaList.pop(0)
            M0 = py * sin(theta1) + px * cos(theta1)
            theta23 = atan2(P * (a2 * cos(theta3) + a3) - M0 * (d4 - a2 * sin(theta3)), 
                            M0 * (a2 * cos(theta3) + a3) + P * (d4 - a2 * sin(theta3)))
            thetaList.append((theta1, theta23 - theta3, theta3))
            
        ################ 求解 theta4 ################
        for index in range(len(thetaList.copy())):
            theta1, theta2, theta3 = thetaList.pop(0)
            c5 = az * cos(theta2 + theta3) - ay * sin(theta1) * sin(theta2 + theta3) - ax * cos(theta1) * sin(theta2 + theta3)
            if abs(c5 - 1) <= 1e-3:
                # 如果 theta5 = 0, theta4 + theta6取决于 theta1, theta2, theta3, 不妨取 theat4 = 0
                theta4 = 0
            else:
                ParamSin = ay * cos(theta1) - ax * sin(theta1)
                ParamCos = az * sin(theta2 + theta3) + (ax * cos(theta1) + ay * sin(theta1)) * cos(theta2 + theta3)
                theta4 = atan2(-ParamSin, -ParamCos)
            thetaList.append((theta1, theta2, theta3, theta4))
            
        ################ 求解 theta5 ################
        for index in range(len(thetaList.copy())):
            theta1, theta2, theta3, theta4 = thetaList.pop(0)
            c5 = az * cos(theta2 + theta3) - ay * sin(theta1) * sin(theta2 + theta3) - ax * cos(theta1) * sin(theta2 + theta3)
            ParamSin = ay * cos(theta1) * sin(theta4) - ax * sin(theta1) * sin(theta4) + \
            az * cos(theta4) * sin(theta2 + theta3) + (ax * cos(theta1) + ay * sin(theta1)) * cos(theta2 + theta3) * cos(theta4)
            theta5 = atan2(-ParamSin, c5)
            thetaList.append((theta1, theta2, theta3, theta4, theta5))
            
        ################ 求解 theta6 ################
        for index in range(len(thetaList.copy())):
            theta1, theta2, theta3, theta4, theta5 = thetaList.pop(0)
            s1, s4, s23 = sin(theta1), sin(theta4), sin(theta2 + theta3)
            c1, c4, c23 = cos(theta1), cos(theta4), cos(theta2 + theta3)
            if abs(theta5) <= 1e-3:
                ParamSin = ny * c1 - nx * s1
                ParamCos = nz * s23 + (ny * s1 + nx * c1) * c23
                theta6 = atan2(ParamSin, ParamCos)
            else:
                ParamSin = c4 * nx * s1 - c1 * c4 * ny + nz * s4 * s23 + (nx * c1 + ny * s1) * s4 * c23
                ParamCos = c4 * ox * s1 - c1 * c4 * oy + oz * s4 * s23 + (ox * c1 + oy * s1) * s4 * c23
                theta6 = atan2(-ParamSin, -ParamCos)
            thetaList.append((theta1, theta2, theta3, theta4, theta5, theta6))
        
        for i in range(len(thetaList.copy())):
            theta1, theta2, theta3, theta4, theta5, theta6 = thetaList[i]
            thetaList.append((theta1, theta2, theta3, theta4 + pi, -theta5, theta6 + pi))
        
        for i in range(len(thetaList.copy())):
            flag = True
            theta = thetaList.pop(0)
            for j in range(len(theta)):
                if not self.limit(theta[j], j):
                    flag = False
                    break
            if flag:
                thetaList.append(theta)
                
        return thetaList
    
    def CalculateParams(self, Q: tuple, W: tuple, A: tuple, T: float = 5) -> tuple:
        """计算5次插值参数

        Args:
            Q (tuple): (start, end) 起点与终点的角度
            W (tuple): (start, end) 起点与终点的角速度
            A (tuple): (start, end) 起点与终点的角加速度
            T (float | int, optional): 从起点到终点经历的时间 Defaults to 5.

        Returns:
            tuple: 参数值 (k0, k1, k2, k3, k4, k5)
        """
        q0, qt = Q
        w0, wt = W
        a0, at = A
        t = T

        k0, k1, k2 = q0, w0, a0/2
        K = at - a0
        M = wt - w0 - a0 * t
        N = qt - q0 - w0 * t - (a0 * t**2) / 2
        k3 = 4 * (5 * N / t - M) / t**2 - (20 * N / (t ** 2) - K) / (2 * t)
        k4 = (20 * N / (t ** 2) - K) / (t ** 2) - 7 * (5 * N / t - M) / (t ** 3)
        k5 = (N / (t ** 3) - k3 - t * k4) / (t ** 2)
        return k0, k1, k2, k3, k4, k5
    
    def traj(self, QStart, QEnd, T: float = 5, gapPoint: int = 50) -> Traj:
        """根据起始状态各关节角与终止状态各关节角规划轨迹

        Args:
            QStart (_type_): 起始关节角
            QEnd (_type_): 终止关节角
            T (float | int, optional): 起始到终止经历时间. Defaults to 5.
            gapPoint (int, optional): 计算 0~T之间times个点的关节角. Defaults to 50.

        Returns:
            np.ndarray: _description_
        """
        K = []
        Trajectory = Traj(len(QStart), gapPoint)
        for i in range(len(QStart)):
            K.append(self.CalculateParams((QStart[i], QEnd[i]), (0, 0), (0, 0), T))
        for k in range(gapPoint):
            t = T / (gapPoint - 1) * k
            Trajectory.t[k] = t
            for i in range(len(QStart)):
                Qt = K[i][0] + K[i][1] * t + K[i][2] * t**2 + K[i][3] * t**3 + K[i][4] * t**4 + K[i][5] * t**5
                Wt = K[i][1] + 2 * K[i][2] * t + 3 * K[i][3] * t**2 + 4 * K[i][4] * t**3 + 5 * K[i][5] * t**4
                At = 2 * K[i][2] + 6 * K[i][3] * t + 12 * K[i][4] * t**2 + 20 * K[i][5] * t**3
                Trajectory.q[k][i] = Qt
                Trajectory.w[k][i] = Wt
                Trajectory.a[k][i] = At
        return Trajectory
        
        
    
if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['simhei'] 
    plt.rcParams['axes.unicode_minus'] = False
    
    puma = Puma560()
    # 运动学正解
    q = np.array([30, 60, 45, 30, 60, 30])
    print(f"theta1={q[0]} thetea2={q[1]} theta3 = {q[2]} theta4={q[3]} theta5={q[4]} theta6={q[5]}\n", puma.forward(np.deg2rad(q)))

    robot = rtb.models.DH.Puma560()
    # 运动学逆解
    pos1 = np.array([0.4, -0.1, 1.0])  # 起始位置
    Sol1 = puma.inverse(np.array(pos1, dtype=np.float64))[0]

    pos2 = np.array([0.6, 0.3, 1.2])  # 终止位置
    Sol2 = puma.inverse(np.array(pos2, dtype=np.float64))[0]

    Sol1 = np.round(np.rad2deg(Sol1), 2)
    Sol2 = np.round(np.rad2deg(Sol2), 2)
    print("Sol1: ")
    print(f"theta1={Sol1[0]} thetea2={Sol1[1]} theta3 = {Sol1[2]} theta4={Sol1[3]} theta5={Sol1[4]} theta6={Sol1[5]}")
    print("Sol2: ")
    print(f"theta1={Sol2[0]} thetea2={Sol2[1]} theta3 = {Sol2[2]} theta4={Sol2[3]} theta5={Sol2[4]} theta6={Sol2[5]}")
    Sol1 = np.deg2rad(Sol1)
    Sol2 = np.deg2rad(Sol2)
    # 验证结果是否正确
    print("根据Sol1得到的正解T1: \n", np.round(puma.forward(Sol1), 2))
    print("根据Sol2得到的正解T2: \n", np.round(puma.forward(Sol2), 2))

    # 计算从Sol1到Sol2的轨迹
    trajectory = puma.traj(Sol1, Sol2)
    # robot.plot(trajectory.q, backend='pyplot', movie="trajectory.gif")
    
    # 画出速度和加速度曲线
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
    title = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
    fig.suptitle("Puma560各关节角角速度与加速度变化曲线")
    for i, ax in enumerate(axes.flat):
        ax.plot(trajectory.t, trajectory.w[:, i], label="角速度")
        ax.plot(trajectory.t, trajectory.a[:, i], label="加速度")
        ax.set_title(title[i])
        ax.set_xlabel("t")
        ax.legend()
        ax.grid(True)
    plt.savefig("速度与加速度变化曲线.png", dpi=300)
