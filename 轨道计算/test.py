import numpy as np
from astropy import units as u
from astropy.time import Time
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.twobody.propagation import CowellPropagator
from poliastro.core.perturbations import J2_perturbation
from poliastro.core.propagation import func_twobody

# 初始轨道
epoch = Time("2025-09-18T00:00:00", scale="utc")
orb = Orbit.from_classical(
    attractor=Earth,
    a=7000e3*u.m, ecc=0.001*u.one,
    inc=51.6*u.deg, raan=247.46*u.deg, argp=130.536*u.deg, nu=10*u.deg,
    epoch=epoch
)
# J2 常数
J2 = Earth.J2.value
R = Earth.R.to(u.km).value
mu = Earth.k.to(u.km**3 / u.s**2).value

def f(t0, u_, k):
    du_kep = func_twobody(t0, u_, k)
    ax, ay, az = J2_perturbation(
        t0, u_, k, J2=Earth.J2.value, R=Earth.R.to(u.km).value
    )
    du_ad = np.array([0, 0, 0, ax, ay, az])
    return du_kep + du_ad

tof = 100 * u.day
# new_orb = orb.propagate(tof, method=CowellPropagator())
new_orb = orb.propagate(tof, method=CowellPropagator(f=f))

print("位置 (km):", new_orb.r.to(u.m).value)
print("速度 (km/s):", new_orb.v.to(u.m/u.s).value)