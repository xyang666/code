import numpy as np
from scipy.integrate import solve_ivp
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord
from astropy.utils import iers
import astropy.units as u
from poliastro.constants import J2_earth, GM_earth, M_earth, R_earth
from OrbitPropagate import propagate_kepler, propagate_numerical
from CoordinateTransfrom import keplerian_to_cartesian, cartesian_to_keplerian

# iers.conf.auto_download = False  # 禁用自动下载
iers.conf.iers_degraded_accuracy = 'warn'   # 仅在使用降级数据时发出警告

def maneuver_Kepler(r0, v0, delta_v, t0, t):
    """
    二体开普勒脉冲机动 (不支持多个目标时刻)
    Parameters:
        r0      : 初始位置矢量 (m)
        v0      : 初始速度矢量 (m/s)
        delta_v : 速度增量 (m/s)
        t0      : 起始时刻 (astropy Time)
        t       : 目标时刻 (astropy Time)
        
    Returns:
        r, v : 在 ECI 下的矢量 (m, m/s)
    """
    v0_new = v0 + delta_v
    kepler = cartesian_to_keplerian(r0, v0_new)
    r, v = propagate_kepler(kepler['a'].value, kepler['e'].value, kepler['i'].value, 
                     kepler['raan'].value, kepler['argp'].value, kepler['nu'].value, 
                     t0, t)
    
    return r, v

def maneuver_J2(r0, v0, delta_v, t0, t):
    """
    J2摄动脉冲机动 (支持多个目标时刻)
    Parameters:
        r0      : 初始位置矢量 (m)
        v0      : 初始速度矢量 (m/s)
        delta_v : 速度增量 (m/s)
        t0      : 起始时刻 (astropy Time)
        t       : 目标时刻 (astropy Time)
        
    Returns:
        r, v : 在 ECI 下的矢量 (m, m/s)
    """
    v0_new = v0 + delta_v
    result = propagate_numerical(r0, v0_new, t0, t, use_j2=True)
    
    return result

if __name__ == "__main__":
    # 示例
    r0 = np.array([7000e3, 0, 0])      # 初始位置 (m)
    v0 = np.array([0, 7.5e3, 1e3])     # 初始速度 (m/s)
    delta_v = np.array([0, 0, 1000])     # 速度增量 (m/s)
    t0 = Time("2025-09-18T00:00:00", scale="utc")
    t = Time("2025-09-18T01:00:00", scale="utc")
    
    # r, v = maneuver_Kepler(r0, v0, delta_v, t0, t)
    result = maneuver_J2(r0, v0, delta_v, t0, t)
    r, v = result['r'][-1, :], result['v'][-1, :]
    print("Final position (m):", r)
    print("Final velocity (m/s):", v)
    
    import poliastro.maneuver as pm
    from poliastro.twobody import Orbit
    from poliastro.bodies import Earth

    orbit = Orbit.from_vectors(Earth, r0 * u.m, v0 * u.m / u.s, epoch=t0)
    man = pm.Maneuver((0 * u.s, delta_v * u.m / u.s))
    orbit = orbit.apply_maneuver(man)
    new_orbit = orbit.propagate(t - t0)
    print("Final position (m):", new_orbit.r.to(u.m))