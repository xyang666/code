import numpy as np
from astropy.coordinates import CartesianRepresentation, ITRS, GCRS
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.utils import iers
import erfa
iers.conf.auto_download = False  # 禁用自动下载

def keplerian_to_cartesian(a, e, i, raan, argp, nu, mu):
    """
    将开普勒轨道根数转换为笛卡尔位置和速度矢量 (IJK坐标系)。

    参数:
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

    返回:
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
    # 注意：代码中使用的是旋转矩阵的转置
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
    
    return coord_trans

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
    iers_a = iers.IERS_Auto.open()

    # 章动+岁差矩阵
    pn = erfa.pnm00a(utc_time.tt.jd1, utc_time.tt.jd2)

    # 地球自转角
    era = erfa.gst00b(utc_time.ut1.jd1, utc_time.ut1.jd1)

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

def eci_to_ecef_CIO(x_eci, y_eci, z_eci, utc_time):
    """"
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
    iers_a = iers.IERS_Auto.open()

    # 章动+岁差矩阵
    x, y, s = erfa.xys06a(utc_time.tt.jd1, utc_time.tt.jd1)
    pn = erfa.c2ixys(x, y, s)
    
    # 地球自转角
    era = erfa.era00(utc_time.ut1.jd1, utc_time.ut1.jd1)

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



if __name__ == "__main__":
    pass