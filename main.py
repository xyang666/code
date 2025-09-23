# Copyright (c) 2025 XIE YANG
# Licensed under the Apache 2.0 License.

import numpy as np
import matplotlib.pyplot as plt
from orbit.utils.OrbitPropagate import propagate_numerical, accel_two_body
from astropy.time import Time
from astropy import units as u
from scipy.integrate import solve_ivp

# ========== PN 导引律 ==========
def pn_accel(r, v, N=3.0):
    """
    3D 矢量形式比例导引律
    r: 相对位置向量 (target - missile)
    v: 相对速度向量 (target - missile)
    N: 导引增益系数
    """
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    r_norm = np.linalg.norm(r)
    if r_norm < 1e-6:
        return np.zeros(3)
    Vc = -np.dot(r, v) / r_norm  # 闭合速度
    return N * Vc * np.cross(np.cross(r, v), r) / (r_norm**3)

# ========== 仿真参数 ==========
dt = 1       # 步长 (s)
T = 6000         # 总时间 (s)
steps = int(T/dt)

# 导弹初始条件
m_pos = np.array([7000e3, 0, 0])
m_vel = np.array([0, 7.5e3, 1e3])

# 目标初始条件
t_pos = np.array([8000e3, 1000e3, 0])
t_vel = np.array([0, 6.5e3, 1e3])

# 轨迹记录
m_traj = [m_pos.copy()]
t_traj = [t_pos.copy()]
'''
t0 = Time.now()
# ========== 仿真循环 ==========
for _ in range(steps):
    r = t_pos - m_pos
    v = t_vel - m_vel
    a_t = accel_two_body(t_pos).value
    a_cmd = pn_accel(r, v, N=3.0)
    # 拓展比例导引（考虑目标加速度横向补偿）
    # a_cmd = pn_accel(r, v, N=3.0) + 0.1*(a_t - (np.dot(a_t, r) / np.dot(r, r)) * r)
    # f = np.linalg.norm(a_cmd - accel_two_body(m_pos).value)

    # 导弹更新
    m_vel = m_vel + a_cmd*dt
    m_pos = m_pos + m_vel*dt

    # 目标更新
    t = t0+dt*u.s
    result = propagate_numerical(t_pos, t_vel, t0, t)
    t_pos = result['r'][-1, :]
    t_vel = result['v'][-1, :]
    t0 = t
    
    m_traj.append(m_pos.copy())
    t_traj.append(t_pos.copy())
    
    if np.linalg.norm(r) < 2000.0:
        print(f"Distance: {np.linalg.norm(r):.2f} m")
        print(f"epoch: {_}")
        break

m_traj = np.array(m_traj)
t_traj = np.array(t_traj)

# ========== 绘图 ==========
plt.figure(figsize=(8,6))
plt.plot(m_traj[:,0], m_traj[:,1], ':', label="Missile")
plt.plot(t_traj[:,0], t_traj[:,1], '--', label="Target")
plt.scatter(m_traj[0,0], m_traj[0,1], c='g', marker='o', label="Missile Start")
plt.scatter(t_traj[0,0], t_traj[0,1], c='r', marker='x', label="Target Start")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("Proportional Navigation (PN) Intercept Example")
plt.legend()
plt.axis("equal")
plt.grid(True)
plt.show()
'''

def rhs(t, y):
    t_pos = y[0:3]
    t_vel = y[3:6]
    m_pos = y[6:9]
    m_vel = y[9:12]
    r = t_pos - m_pos
    v = t_vel - m_vel
    t_a = accel_two_body(t_pos).value
    m_a = pn_accel(r, v, N=3.0)
    
    return np.hstack((t_vel, t_a, m_vel, m_a))

def event(t, y):
    t_pos = y[0:3]
    m_pos = y[6:9]
    r = t_pos - m_pos
    return not (np.linalg.norm(r) < 1.0)

event.terminal = True

sol = solve_ivp(rhs, (1, T), np.hstack((t_pos, t_vel, m_pos, m_vel)), 
          t_eval=np.arange(1, T, dt), rtol=1e-9, atol=1e-12, method='RK45', events=event)

# print(f"Final Distance: {np.linalg.norm(sol.y[0:3,-1]-sol.y[6:9,-1]):.2f} m")
print(f"event time: {sol.t_events[0][0]} s")
print(f"Final Distance: {np.linalg.norm(sol.y_events[0][0][0:3]-sol.y_events[0][0][6:9]):.2f} m")