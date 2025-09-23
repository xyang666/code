# Copyright (c) 2025 XIE YANG
# Licensed under the Apache 2.0 License.

import erfa
import numpy as np
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord
from astropy.utils import iers
import astropy.units as u
from poliastro.constants import GM_earth


# iers.conf.auto_download = False  # 禁用自动下载
iers.conf.iers_degraded_accuracy = 'warn'   # 仅在使用降级数据时发出警告

def kepler_to_cartesian(a, e, i, raan, argp, nu, mu=GM_earth):
    """
    将开普勒轨道根数转换为笛卡尔位置和速度矢量 (IJK坐标系)。

    Parameters:
        a : float
            半长轴 (m)
        e : float
            偏心率
        i : float
            轨道倾角 (rad)
        raan : float
            升交点赤经 (rad)
        argp : float
            近地点幅角 (rad)
        nu : float
            真近点角 (rad)
        mu : float
            中心天体重力参数 (m^3/s^2)

    Returns:
        r_ijk : numpy array
            位置矢量 [X, Y, Z] (m)
        v_ijk : numpy array
            速度矢量 [Vx, Vy, Vz] (m/s)
    """

    # 1. 计算比角动量 h 和瞬时距离 r
    h = np.sqrt(mu * a * (1 - e**2))
    r = (h**2 / mu) / (1 + e * np.cos(nu))

    # 2. 在PQW轨道平面坐标系中的位置和速度
    r_pqw = np.array([r * np.cos(nu), r * np.sin(nu), 0])
    v_pqw = np.array([-(mu/h) * np.sin(nu), (mu/h) * (e + np.cos(nu)), 0])

    # 3. 构建旋转矩阵 (从PQW到IJK)
    # 此处为向量旋转矩阵，与坐标系变换矩阵互为转置
    R11 = np.cos(raan)*np.cos(argp) - np.sin(raan)*np.sin(argp)*np.cos(i)
    R12 = -np.cos(raan)*np.sin(argp) - np.sin(raan)*np.cos(argp)*np.cos(i)
    R13 = np.sin(raan)*np.sin(i)
    
    R21 = np.sin(raan)*np.cos(argp) + np.cos(raan)*np.sin(argp)*np.cos(i)
    R22 = -np.sin(raan)*np.sin(argp) + np.cos(raan)*np.cos(argp)*np.cos(i)
    R23 = -np.cos(raan)*np.sin(i)
    
    R31 = np.sin(argp)*np.sin(i)
    R32 = np.cos(argp)*np.sin(i)
    R33 = np.cos(i)
    
    rot_matrix = np.array([[R11, R12, R13],
                           [R21, R22, R23],
                           [R31, R32, R33]])

    # 4. 应用旋转，得到IJK坐标系中的矢量
    r_ijk = rot_matrix @ r_pqw
    v_ijk = rot_matrix @ v_pqw

    return r_ijk, v_ijk

def cartesian_to_kepler(r, v, mu=GM_earth):
    """将eci坐标系(笛卡尔系)位置 r 和速度 v 转换为 Kepler 元素。

    Parameters:
    r: array-like Quantity (3,) -- 位置向量(m)
    v: array-like Quantity (3,) -- 速度向量(m/s)
    mu: astropy Quantity -- 引力参数，默认地球值

    Returns:
    dict 包含 a, e, i, raan, argp, nu
    """
    # 规范化单位到 m, m/s
    mu = mu.to(u.m**3 / u.s**2).value

    r_norm = np.linalg.norm(r)
    v_norm = np.linalg.norm(v)

    # 比角动量 h
    h = np.cross(r, v)
    h_norm = np.linalg.norm(h)

    # 节点向量 n = k x h
    k_vec = np.array([0.0, 0.0, 1.0])
    n = np.cross(k_vec, h)
    n_norm = np.linalg.norm(n)

    # 偏心率向量
    e_vec = (np.cross(v, h) / mu) - (r / r_norm)
    e = np.linalg.norm(e_vec)

    # 比能量
    energy = v_norm**2 / 2.0 - mu / r_norm

    # 半长轴 a（处理抛物线 a = inf）
    if np.isclose(energy, 0.0, atol=1e-12):
        a = np.inf
    else:
        a = -mu / (2.0 * energy)

    # 倾角 i
    i = np.arccos(h[2] / h_norm)

    # 升交点赤经 RAAN（Ω）
    if n_norm != 0.0:
        raan = np.arccos(n[0] / n_norm)
        if n[1] < 0:
            raan = 2*np.pi - raan
    else:
        # 赤道轨道，RAAN 未定义（设为 0）
        raan = 0.0

    # 近地点幅角 argp（ω）
    if n_norm != 0.0 and e > 1e-12:
        argp = np.arccos(np.dot(n, e_vec) / (n_norm * e))
        if e_vec[2] < 0:
            argp = 2*np.pi - argp
    else:
        # 圆轨道或赤道轨道时，argp 未定义或用其他量替代
        argp = 0.0

    # 真近点角 ν
    if e > 1e-12:
        nu = np.arccos(np.dot(e_vec, r) / (e * r_norm))
        if np.dot(r, v) < 0:
            nu = 2*np.pi - nu
    else:
        # 近圆轨道：用结点向量或 x 方向来定义真近点角
        if n_norm != 0.0:
            # 用结点向量 n
            cos_nu = np.dot(n, r) / (n_norm * r_norm)
            cos_nu = np.clip(cos_nu, -1.0, 1.0)
            nu = np.arccos(cos_nu)
            if r[2] < 0:
                nu = 2*np.pi - nu
        else:
            # 赤道且近圆：退化情形，直接以 x 投影定义
            cos_nu = r[0] / r_norm
            cos_nu = np.clip(cos_nu, -1.0, 1.0)
            nu = np.arccos(cos_nu)
            if r[1] < 0:
                nu = 2*np.pi - nu

    # 单位转换并打包结果
    res = {
        'a': (a * u.m) if np.isfinite(a) else np.inf,
        'e': e * u.dimensionless_unscaled,
        'i': i * u.rad,
        'raan': raan * u.rad,
        'argp': argp * u.rad,
        'nu': nu * u.rad,
        'h_vec': (h * (u.m**2 / u.s)),
        'e_vec': (e_vec * u.dimensionless_unscaled)
    }

    return res

def eci_to_ecef(x_eci, y_eci, z_eci, utc_time):
    """
    将地心惯性坐标系 (ECI) 的坐标转换为地心地固坐标系 (ECEF)。
    Parameters:
        x_eci (float): ECI坐标系下的X坐标 (米)。
        y_eci (float): ECI坐标系下的Y坐标 (米)。
        z_eci (float): ECI坐标系下的Z坐标 (米)。
        utc_time (str 或 astropy.time.Time): 与ECI坐标对应的UTC时间，可以是字符串或astropy的Time对象。
    Returns:
        numpy.ndarray: 包含ECEF坐标系下[x, y, z]的三元素数组 (米)。
    Notes:
        - 本函数通过格林尼治视恒星时 (GAST) 计算ECI到ECEF的旋转。
        - 依赖 astropy 和 numpy 库。
    """
    # 格林尼治视恒星时
    t = Time(utc_time, scale='utc')
    gast = t.sidereal_time('apparent', 'greenwich').radian
    
    # 构建旋转矩阵
    Rz = np.array([
        [np.cos(gast), np.sin(gast), 0],
        [-np.sin(gast), np.cos(gast), 0],
        [0, 0, 1]
    ])
    
    # 应用旋转
    ecef = Rz @ np.array([x_eci, y_eci, z_eci])
    return ecef

def eci_to_ecef_acc(x_eci, y_eci, z_eci, utc_time):
    """
    将地心惯性坐标系 (ECI/GCRS) 的位置坐标转换为地心地固坐标系 (ECEF/ITRS)。
    Parameters:
        x_eci (float): ECI坐标系下的X坐标 (米)。
        y_eci (float): ECI坐标系下的Y坐标 (米)。
        z_eci (float): ECI坐标系下的Z坐标 (米)。
        utc_time (datetime 或 astropy.time.Time): 坐标对应的UTC时间。
    Returns:
        astropy.coordinates.SkyCoord: ECEF (ITRS) 坐标系下的位置，SkyCoord对象。
    """
    # 创建 GCRS 坐标（J2000.0 历元）
    coord = SkyCoord(
        x=x_eci, y=y_eci, z=z_eci,
        frame='gcrs',
        representation_type='cartesian',
        obstime=utc_time
    )
    
    # 转换为 ITRS
    coord_trans = coord.transform_to('itrs')
    
    return coord_trans.cartesian.xyz.value

def eci_to_ecef_equinox(x_eci, y_eci, z_eci, utc_time):
    """"
    使用基于春分点的变换（包括岁差、章动、地球自转和极移修正）将地心惯性坐标系 (ECI) 转换为地心地固坐标系 (ECEF)。
    Parameters:
        x_eci : float
            ECI坐标系下的X坐标（米）。
        y_eci : float
            ECI坐标系下的Y坐标（米）。
        z_eci : float
            ECI坐标系下的Z坐标（米）。
        utc_time : astropy.time.Time
            表示变换历元的UTC时间对象。
    Returns:
        ecef : numpy.ndarray
            转换后的ECEF坐标，三元素数组（米）。
    Notes:
        - 使用IERS数据进行极移修正。
        - 应用岁差-章动矩阵、地球自转角和极移矩阵。
        - 极移采用小角度近似。
    """
    iers_a = iers.IERS_Auto.read("./finals2000A.all")

    # 章动+岁差矩阵
    pn = erfa.pnm00a(utc_time.tt.jd1, utc_time.tt.jd2)

    # 地球自转角
    era = erfa.gst00b(utc_time.ut1.jd1, utc_time.ut1.jd2)

    R = np.array([
        [ np.cos(era), np.sin(era), 0.0],
        [-np.sin(era), np.cos(era), 0.0],
        [ 0.0,          0.0,        1.0]
    ])

    # 极移矩阵
    # 获取极移参数
    x_p, y_p = iers_a.pm_xy(utc_time.jd1, utc_time.jd2)

    # 小角度近似
    xp = x_p.to(u.rad).value
    yp = y_p.to(u.rad).value

    # s' 修正
    sp = erfa.sp00(utc_time.tt.jd1, utc_time.tt.jd2)
    pom = erfa.pom00(xp, yp, sp)

    # 合成总矩阵
    total = pom @ R @ pn
    ecef = total @ np.array([x_eci, y_eci, z_eci])
    return ecef

def ecef_to_eci_equinox(x_ecef, y_ecef, z_ecef, utc_time):
    """"
    使用基于春分点的变换（包括岁差、章动、地球自转和极移修正）将地心地固坐标系(ECEF)转换为地心惯性坐标系(ECI)。
    """
    iers_a = iers.IERS_Auto.read("./finals2000A.all")

    # 章动+岁差矩阵
    pn = erfa.pnm00a(utc_time.tt.jd1, utc_time.tt.jd2)

    # 地球自转角
    era = erfa.gst00b(utc_time.ut1.jd1, utc_time.ut1.jd2)

    R = np.array([
        [ np.cos(era), np.sin(era), 0.0],
        [-np.sin(era), np.cos(era), 0.0],
        [ 0.0,          0.0,        1.0]
    ])

    # 极移矩阵
    # 获取极移参数
    x_p, y_p = iers_a.pm_xy(utc_time.jd1, utc_time.jd2)

    # 小角度近似
    xp = x_p.to(u.rad).value
    yp = y_p.to(u.rad).value

    # s' 修正
    sp = erfa.sp00(utc_time.tt.jd1, utc_time.tt.jd2)
    pom = erfa.pom00(xp, yp, sp)

    # 合成总矩阵
    total = pom @ R @ pn
    eci = total.T @ np.array([x_ecef, y_ecef, z_ecef])
    return eci

def eci_to_ecef_CIO(x_eci, y_eci, z_eci, utc_time):
    """
    使用基于CIO的变换（包括岁差、章动、地球自转和极移修正）将地心惯性坐标系 (ECI) 转换为地心地固坐标系 (ECEF)。
    Parameters:
        x_eci : float
            ECI坐标系下的X坐标（米）。
        y_eci : float
            ECI坐标系下的Y坐标（米）。
        z_eci : float
            ECI坐标系下的Z坐标（米）。
        utc_time : astropy.time.Time
            表示变换历元的UTC时间对象。
    Returns:
        ecef : numpy.ndarray
            转换后的ECEF坐标，三元素数组（米）。
    Notes:
        - 使用IERS数据进行极移修正。
        - 应用岁差-章动矩阵、地球自转角和极移矩阵。
        - 极移采用小角度近似。
    """
    iers_a = iers.IERS_Auto.read("./finals2000A.all")

    # 章动+岁差矩阵
    x, y, s = erfa.xys06a(utc_time.tt.jd1, utc_time.tt.jd2)
    pn = erfa.c2ixys(x, y, s)
    
    # 地球自转角
    era = erfa.era00(utc_time.ut1.jd1, utc_time.ut1.jd2)
    # 使用简化公式计算 ERA
    # tu = utc_time.ut1.jd - 2451545.0
    # era = 2* np.pi * ((0.7790572732640 + 1.00273781191135448 * tu) % 1)

    R = np.array([
        [ np.cos(era), np.sin(era), 0.0],
        [-np.sin(era), np.cos(era), 0.0],
        [ 0.0,          0.0,        1.0]
    ])

    # 极移矩阵
    # 获取极移参数
    x_p, y_p = iers_a.pm_xy(utc_time.jd1, utc_time.jd2)

    # 小角度近似
    xp = x_p.to(u.rad).value
    yp = y_p.to(u.rad).value

    # s' 修正
    sp = erfa.sp00(utc_time.tt.jd1, utc_time.tt.jd2)
    pom = erfa.pom00(xp, yp, sp)

    # 合成总矩阵
    total = pom @ R @ pn
    ecef = total @ np.array([x_eci, y_eci, z_eci])
    return ecef

def ecef_to_eci_CIO(x_ecef, y_ecef, z_ecef, utc_time):
    """
    使用基于CIO的变换（包括岁差、章动、地球自转和极移修正）将地心地固坐标系 (ECEF) 转换为地心惯性坐标系 (ECI) 。
    """
    iers_a = iers.IERS_Auto.read("./finals2000A.all")

    # 章动+岁差矩阵
    x, y, s = erfa.xys06a(utc_time.tt.jd1, utc_time.tt.jd2)
    pn = erfa.c2ixys(x, y, s)
    
    # 地球自转角
    era = erfa.era00(utc_time.ut1.jd1, utc_time.ut1.jd2)

    R = np.array([
        [ np.cos(era), np.sin(era), 0.0],
        [-np.sin(era), np.cos(era), 0.0],
        [ 0.0,          0.0,        1.0]
    ])

    # 极移矩阵
    # 获取极移参数
    x_p, y_p = iers_a.pm_xy(utc_time.jd1, utc_time.jd2)

    # 小角度近似
    xp = x_p.to(u.rad).value
    yp = y_p.to(u.rad).value

    # s' 修正
    sp = erfa.sp00(utc_time.tt.jd1, utc_time.tt.jd2)
    pom = erfa.pom00(xp, yp, sp)

    # 合成总矩阵
    total = pom @ R @ pn
    eci = total.T @ np.array([x_ecef, y_ecef, z_ecef])
    return eci

if __name__ == "__main__":
    # # 初始轨道根数
    # a = 7000e3  # m
    # e = 0.001
    # inc = np.radians(51.6)
    # raan = np.radians(247.46)
    # argp = np.radians(130.536)
    # nu0 = np.radians(10.0)

    # r0, v0 = kepler_to_cartesian(a, e, inc, raan, argp, nu0, GM_earth.value)
    # t0 = Time("2025-09-18T00:00:00", scale="utc")
    # # t_targets = t0 + TimeDelta(np.arange(0, 600, 60)*u.s)
    # t_targets = t0 + 1*u.day
    
    
    from poliastro.twobody import Orbit
    from poliastro.bodies import Earth
    epoch = Time("2025-09-18T00:00:00", scale="utc")
    orb = Orbit.from_classical(
        attractor=Earth,
        a=7000e3*u.m, ecc=0.001*u.one,
        inc=51.6*u.deg, raan=247.46*u.deg, argp=130.536*u.deg, nu=10*u.deg,
        epoch=epoch
    )
    r0v0 = orb._state.to_vectors()
    r0 = r0v0.r.to_value(u.m)
    v0 = r0v0.v.to_value(u.m/u.s)
    k = cartesian_to_kepler(r0, v0)
    
    # res = propagate_numerical(r0, v0, t0, t_targets, use_j2=False, atol=1e-12, rtol=1e-11)
    # r, v = propagate_kepler(a, e, inc, raan, argp, nu0, t0, t_targets)
    # print(f"Kepler Propagation (r:{r}, v:{v})")
    # print(f"Numerical Propagation (r:{res['r'][-1]}, v:{res['v'][-1]})")
    print(f"r0:{r0}, v0:{v0}")
    
    # t = Time("2025-09-18T00:00:00", scale="utc")

    # ecef = eci_to_ecef_CIO(1,0,0, t)
    # print("CIO:{}".format(ecef))    

    # ecef = eci_to_ecef_equinox(1,0,0, t)
    # print("equinox:{}".format(ecef))
    
    # ecef = eci_to_ecef(1,0,0, t)
    # print("astropy:{}".format(ecef))
    
    # ecef = eci_to_ecef_acc(1,0,0, t)
    # print("astropy_acc:{}".format(ecef))