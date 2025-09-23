import numpy as np
from scipy.integrate import solve_ivp
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord
from astropy.utils import iers
import astropy.units as u
from poliastro.constants import J2_earth, GM_earth, M_earth, R_earth

# iers.conf.auto_download = False  # 禁用自动下载
iers.conf.iers_degraded_accuracy = 'warn'   # 仅在使用降级数据时发出警告

def solve_kepler(M, e):
    """
    解开普勒方程 M = E - e*sin(E)，求偏近点角 E。

    Parameters:
        M (float): 平近点角（弧度）。
        e (float): 轨道偏心率（0 <= e < 1）。

    Returns:
        float: 对应于给定平近点角和偏心率的偏近点角 E（弧度）。

    Notes:
        使用牛顿迭代法求解方程：
            E - e*sin(E) = M
        迭代次数为 50 次，通常可收敛。
    """
    E = M
    for _ in range(50):
        E = E - (E - e*np.sin(E) - M)/(1 - e*np.cos(E))
    return E

def propagate_kepler(a, e, i, raan, argp, nu0, t0, t):
    """
    二体开普勒传播 (椭圆轨道)(不支持多个目标时刻)
    Parameters:
        a    : 半长轴 (m)
        e    : 偏心率
        i    : 倾角 (rad)
        raan : 升交点赤经 (rad)
        argp : 近地点幅角 (rad)
        nu0  : 初始真近点角 (rad)
        t0   : 起始时刻 (astropy Time)
        t    : 目标时刻 (astropy Time)
        
    Returns:
        r, v : 在 ECI 下的矢量 (m, m/s)
    """

    # 轨道参数
    n = np.sqrt(GM_earth.value / a**3)   # 平均运动 rad/s

    # 初始真近点角 -> 偏近点角 E0
    E0 = 2*np.arctan(np.sqrt((1-e)/(1+e)) * np.tan(nu0/2))
    M0 = E0 - e*np.sin(E0)         # 初始平近点角

    # 时间差
    dt = (t - t0).to(u.s).value
    M = M0 + n*dt                  # 目标平近点角

    # 解开普勒方程 M = E - e*sinE
    E = solve_kepler(M, e)

    # 偏近点角 -> 真近点角
    nu = 2*np.arctan2(np.sqrt(1+e)*np.sin(E/2),
                      np.sqrt(1-e)*np.cos(E/2))

    # 距离
    p = a*(1 - e**2)
    r_norm = p/(1 + e*np.cos(nu))

    # 在轨道平面 PQW
    r_pf = np.array([r_norm*np.cos(nu), r_norm*np.sin(nu), 0])
    v_pf = np.sqrt(GM_earth/p) * np.array([-np.sin(nu), e+np.cos(nu), 0])

    # PQW -> ECI
    # 此处为向量旋转矩阵，与坐标系变换矩阵互为转置
    cosO, sinO = np.cos(raan), np.sin(raan)
    cosi, sini = np.cos(i), np.sin(i)
    cosw, sinw = np.cos(argp), np.sin(argp)
    R = np.array([
        [cosO*cosw - sinO*sinw*cosi, -cosO*sinw - sinO*cosw*cosi, sinO*sini],
        [sinO*cosw + cosO*sinw*cosi, -sinO*sinw + cosO*cosw*cosi, -cosO*sini],
        [sinw*sini,                  cosw*sini,                   cosi]
    ])

    r = R @ r_pf
    v = R @ v_pf
    return r, v

def accel_two_body(r_vec, mu=GM_earth):
    """纯二体加速度（m/s^2）"""
    r = np.linalg.norm(r_vec)
    return -mu * r_vec / r**3

def accel_j2(r_vec, mu=GM_earth, J2_val=J2_earth, R_e=R_earth):
    """
    二体 + J2 摄动加速度（近地常用项）。
    r_vec: m
    返回加速度 m/s^2
    公式参考：标准 J2 加速度项
    """
    r = np.linalg.norm(r_vec)
    x, y, z = r_vec
    zx = z / r
    factor = 1.5 * J2_val * mu * (R_e.value**2) / r**5

    ax = -mu * x / r**3 + factor * x * (5 * zx**2 - 1)
    ay = -mu * y / r**3 + factor * y * (5 * zx**2 - 1)
    az = -mu * z / r**3 + factor * z * (5 * zx**2 - 3)
    return np.array([ax, ay, az])

def rhs(t, y, mu, use_j2=False):
    """
    ODE RHS
    
    Parameters:
    y = [x,y,z, vx,vy,vz]
    t: seconds (unused for autonomous but required by solver)

    Returns:
    dydt = [vx,vy,vz, ax,ay,az]
    """
    r = y[0:3]
    v = y[3:6]
    if use_j2:
        a = accel_j2(r, mu=mu)
    else:
        a = accel_two_body(r, mu=mu)
    return np.hstack((v, a))

def propagate_numerical(r0, v0, t0, t_targets, mu=GM_earth, use_j2=False,
                        rtol=1e-9, atol=1e-12, method='DOP853'):
    """
    数值积分轨道（支持多个目标时刻）

    Parameters
    ----------
    r0 : array-like, shape (3,)  初始位置 [m]
    v0 : array-like, shape (3,)  初始速度 [m/s]
    t0 : astropy.time.Time        初始时刻
    t_targets : astropy.time.Time or array-like of astropy.time.Time
        目标时刻（可以是单个 Time 或 TimeArray）
    mu : float
    use_j2 : bool  是否包含 J2 摄动
    rtol, atol : solver 误差控制
    method : str   solve_ivp 方法（'DOP853' 推荐用于高精度）
    
    Returns
    -------
    results : dict
      {
        'times': numpy array of astropy Time (same order as t_targets sorted),
        'r': ndarray (N,3) m,
        'v': ndarray (N,3) m/s
      }
      
    Notes
    -----
    - solver 在内部以秒为单位积分； t_targets 会被转换为以秒为参考 t0 的偏移秒数。
    - 若 t_targets 等于 t0，会返回初始状态对应点（不积分）。
    """
    
    # 规范化 t_targets 到数组
    if isinstance(t_targets, Time):
        t_targets_arr = t_targets if t_targets.shape != () else Time([t_targets])
    else:
        t_targets_arr = Time(t_targets)

    # 计算目标时刻相对于 t0 的秒偏移，并排序（solver 要求 t_eval 单调）
    dt_targets = (t_targets_arr - t0).to(u.s).value
    dt_targets = np.append(dt_targets, 0.0)  # 确保包含初始时刻
    # 去除 dt_targets 中的重复元素（避免 solve_ivp t_eval 重复导致报错）
    _, unique_idx = np.unique(dt_targets, return_index=True)
    dt_targets = dt_targets[unique_idx]
    order = np.argsort(dt_targets)
    dt_sorted = dt_targets[order]

    # 初始状态向量
    y0 = np.hstack((np.asarray(r0, dtype=float), np.asarray(v0, dtype=float)))

    # 如果所有目标时刻都等于初始时刻，直接返回
    if np.allclose(dt_sorted, 0.0):
        r_out = np.tile(y0[:3], (len(dt_sorted), 1))
        v_out = np.tile(y0[3:], (len(dt_sorted), 1))
        return {
            'times': t_targets_arr[order],
            'r': r_out,
            'v': v_out
        }

    # solver: 从 min(dt_sorted) 到 max(dt_sorted) 积分，但常见情况是 dt_sorted[0] >= 0 或部分为负（向后传播）
    t_min = float(np.min(dt_sorted))
    t_max = float(np.max(dt_sorted))

    # solve_ivp 要求 t_span 有序 (t0_span, t1_span)
    # 这里假设目标时刻在同一方向
    t_span = (t_min, t_max)

    sol = solve_ivp(fun=lambda tt, yy: rhs(tt, yy, mu=GM_earth.value, use_j2=use_j2),
                    t_span=t_span,
                    y0=y0,
                    method=method,
                    t_eval=dt_sorted,
                    rtol=rtol,
                    atol=atol)

    if not sol.success:
        raise RuntimeError("数值积分失败: " + str(sol.message))

    r_sol = sol.y[0:3, :].T  # shape (N,3)
    v_sol = sol.y[3:6, :].T

    # 还原为输入顺序
    r_out = r_sol[order.argsort()]
    v_out = v_sol[order.argsort()]

    return {
        'times': t_targets_arr,
        'r': r_out,
        'v': v_out
    }