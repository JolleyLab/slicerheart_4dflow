"""
Synthetic 4D flow phantoms with analytic ground truth, for tests and
continuous integration (no data download).

Every phantom returns a dict with the clean velocity field, the corrupted
measurement, magnitude images, lumen mask, voxel_to_world and the analytic
quantities needed to score a filter (true offset field, wrap counts,
pressure, wall shear stress where defined). All velocities are in mm/s and
follow the package conventions: arrays [t, k, j, i, 3] with world
components (vx, vy, vz); see geometry.py.

Phantoms
--------
poiseuille()   Steady or pulsatile Hagen-Poiseuille flow in a straight
               cylinder. Exact pressure gradient and wall shear stress:
               for parabolic profile v(r) = v_max (1 - r^2/R^2),
               dp/dz = -4 mu v_max / R^2 and WSS = 2 mu v_max / R.
               Reference: Batchelor GK. An Introduction to Fluid Dynamics.
               Cambridge University Press, 1967 (reissued 2000).
               doi:10.1017/CBO9780511800955.

womersley()    Pulsatile flow in a straight cylinder driven by a sinusoidal
               pressure gradient, the exact unsteady solution used to
               validate 4D flow pressure and WSS methods.
               Reference: Womersley JR. Method for the calculation of
               velocity, rate of flow and viscous drag in arteries when the
               pressure gradient is known. J Physiol 1955;127(3):553-563.
               PMID 14368548. doi:10.1113/jphysiol.1955.sp005276.

stenosis()     Axisymmetric tube with a cosine constriction (a coarctation
               or valvar stenosis model). The field is the lubrication
               (slowly varying tube) approximation: parabolic axial profile
               scaled to the local radius, radial velocity from continuity,
               so it is exactly divergence-free, conserves flow at every
               section, and satisfies no-slip. Analytic: throat mean and
               peak velocity, centreline pressure from the axial momentum
               balance (inertial Bernoulli term plus Poiseuille viscous
               term), simplified-Bernoulli gradient, and wall shear stress
               proportional to Q / R(z)^3. The jet exceeds a clinical venc
               throughout the cycle, which is the case temporal unwrapping
               cannot handle.
               References: Young DF, Tsai FY. Flow characteristics in
               models of arterial stenoses. I. Steady flow. J Biomech
               1973;6(4):395-410. PMID 4732939.
               doi:10.1016/0021-9290(73)90099-7 (cosine stenosis geometry,
               pressure drop). Hatle L, Brubakk A, Tromsdal A, Angelsen B.
               Noninvasive assessment of pressure drop in mitral stenosis
               by Doppler ultrasound. Br Heart J 1978;40(2):131-140.
               PMID 637964. doi:10.1136/hrt.40.2.131 (simplified
               Bernoulli).

bifurcation()  A parent vessel dividing into two daughters at a set angle
               with a set flow split, each branch carrying Poiseuille flow
               and the daughter radii following Murray's law
               (R0^3 = R1^3 + R2^3 with flow proportional to R^3). Away
               from the junction every branch is exact; at the junction the
               field is a union of the three cylinders and is not
               divergence-free, which is marked by a `junction` mask.
               Returns measurement planes in each branch so that
               mass-conservation and flow-split (Qp:Qs type) checks have a
               known answer.
               References: Murray CD. The physiological principle of
               minimum work. I. The vascular system and the cost of blood
               volume. Proc Natl Acad Sci USA 1926;12(3):207-214.
               PMID 16576980. doi:10.1073/pnas.12.3.207.
               Bissell MM, Raimondi F, Ait Ali L, et al. 4D Flow
               cardiovascular magnetic resonance consensus statement: 2023
               update. J Cardiovasc Magn Reson 2023;25(1):40.
               PMID 37474977 (quality assurance).

valve_jet()    A regurgitant or stenotic valve jet: an orifice jet entering
               a chamber of slow swirling blood, present for a few frames
               only. Jet: Gaussian cross-profile whose radius widens
               linearly with distance (free-jet spreading) and whose
               centreline velocity decays beyond a potential core; chamber:
               rigid-body swirl (divergence-free). The field is kinematic
               (a known answer for unwrapping and multi-venc tests, not a
               solution of the flow equations). Returns an intravoxel
               velocity-standard-deviation map (turbulence intensity times
               the jet speed in the shear layer and the decaying core) for
               the dephasing model of acquire()/acquire_multivenc(), the
               orifice flow per frame and the regurgitant volume, and
               region masks (jet core, jet edge, chamber).
chest()        A thorax-like phantom with static tissue, two lungs, a
               pulsatile artery, a slow steady vein, a second-order
               background phase offset and an optional phase-encode
               wrap-around band. Used to test background phase correction
               and noise masking. Offset magnitudes follow the multi-centre
               survey of Gatehouse PD, Rolf MP, Graves MJ, et al. Flow
               measurement by cardiovascular magnetic resonance: a
               multi-centre multi-vendor study of background phase offset
               errors that can compromise the accuracy of derived
               regurgitant or shunt flow measurements. J Cardiovasc Magn
               Reson 2010;12:5. PMID 20074359.

Acquisition model
-----------------
acquire()      Phase-contrast forward model (intravoxel_std: see below). Each velocity component is the
               phase of a complex signal, S_c = M exp(i pi v_c / venc_c),
               measured against an unencoded reference S_0 = M (simple
               four-point encoding), with independent complex Gaussian noise
               on every acquisition. The decoded velocity is
               venc angle(S_c conj(S_0)) / pi. This produces, without any
               further modelling: the velocity noise law
               sigma_v = sqrt(2) venc / (pi SNR) in voxels with signal,
               uniformly random velocity where there is no signal (air,
               lung), aliasing wherever |v| > venc, and, through
               block_average() of the complex signals, intravoxel averaging
               (partial volume) at the vessel wall. The noise law follows
               from first principles: phase noise 1/SNR per acquisition,
               sqrt(2)/SNR for the difference of two, scaled by venc/pi.
               References:
               Pelc NJ, Bernstein MA, Shimakawa A, Glover GH. Encoding
               strategies for three-direction phase-contrast MR imaging of
               flow. J Magn Reson Imaging 1991;1(4):405-413. PMID 1790362.
               doi:10.1002/jmri.1880010404. (Four-point encoding.)
               Bernstein MA, King KF, Zhou XJ. Handbook of MRI Pulse
               Sequences. Elsevier Academic Press, 2004.
               ISBN 9780120928613.
               Dyverfeldt P, Bissell M, Barker AJ, et al. 4D flow
               cardiovascular magnetic resonance consensus statement.
               J Cardiovasc Magn Reson 2015;17:72. PMID 26257141.
               doi:10.1186/s12968-015-0174-5.
block_average() Intravoxel averaging of complex signals (partial volume) to
               a coarser grid; the correct way to downsample phase-contrast
               data, as opposed to averaging velocities.
acquire_multivenc() The same field acquired at several vencs, as in dual-
               or triple-venc 4D flow: either interleaved with one shared
               reference acquisition (Schnell 2017; Ma 2019; the dual-venc
               5D flow of Weiss, Roy et al. 2024), so the reference noise is
               common to all vencs, or as separate scans with independent
               noise (Franco 2022). Input for multivenc.multivenc_unwrap.
               Two further mechanisms of real jets: intravoxel_std, a map
               of the velocity spread within a voxel, attenuates each
               encoded signal by exp(-pi^2 sigma_v^2 / (2 venc^2)), the
               magnitude loss of a Gaussian intravoxel velocity
               distribution (Dyverfeldt et al. 2006), which is stronger at
               low venc; and time_offset_frames, one per venc, samples the
               field at frame times shifted by a fraction of a frame
               (Ma 2019 attribute systolic unwrapping errors to timing
               differences between the interleaved encodings).
               Dyverfeldt P, Sigfridsson A, Kvitting JP, Ebbers T.
               Quantification of intravoxel velocity standard deviation and
               turbulence intensity by generalizing phase-contrast MRI.
               Magn Reson Med 2006;56(4):850-858. doi:10.1002/mrm.21022.

Corruption helpers (simpler alternatives to acquire())
------------------
add_noise()    Gaussian velocity noise as a fraction of venc, inside a mask.
alias()        Wrap velocities into [-venc, venc), returning the integer
               wrap count that an unwrapping method must recover.
"""

from __future__ import annotations

import numpy as np
from scipy import special

from .geometry import voxel_grid_world_points, voxel_to_world_matrix

BLOOD_VISCOSITY_PA_S = 3.5e-3      # Pa s, Newtonian approximation (literature: 3.2 to 4.0 mPa s)
BLOOD_DENSITY_KG_M3 = 1060.0


def _cylinder_geometry(shape, voxel_to_world, radius_mm, axis_world, centre_world=None):
    """Radial distance (mm) from the cylinder axis and the axial coordinate
    of every voxel, in world space."""
    pts = voxel_grid_world_points(shape, voxel_to_world)
    n = np.asarray(axis_world, float)
    n = n / np.linalg.norm(n)
    c = pts.reshape(-1, 3).mean(0) if centre_world is None else np.asarray(centre_world, float)
    d = pts - c
    z = d @ n
    r = np.linalg.norm(d - z[..., None] * n, axis=-1)
    return r, z, n


def _waveform(nt, waveform):
    """Frame weights: ones for nt == 1, the aortic waveform, or the given
    waveform, whose length must equal nt."""
    if waveform is None:
        return np.ones(1) if nt == 1 else aortic_waveform(nt)
    w = np.asarray(waveform, float)
    if w.shape[0] != nt:
        raise ValueError(f"waveform has {w.shape[0]} frames but nt={nt}")
    return w


def aortic_waveform(nt):
    """Systolic pulse plus a small diastolic wave, peak about 1."""
    t = np.arange(nt) / nt
    return 0.1 + 0.9 * np.exp(-((t - 0.25) / 0.09) ** 2) + 0.15 * np.exp(-((t - 0.55) / 0.08) ** 2)


def poiseuille(shape=(32, 48, 48), voxel_to_world=None, radius_mm=12.0, v_max_mm_s=800.0,
               axis_world=(0.0, 0.0, 1.0), nt=1, waveform=None, viscosity=BLOOD_VISCOSITY_PA_S):
    """
    Returns dict: velocity (nt, nk, nj, ni, 3) mm/s, mask, voxel_to_world,
    pressure_gradient_pa_mm (per frame, along axis_world), wss_pa (scalar,
    per frame), radius_mm, axis_world, r_mm (radial coordinate per voxel).
    With nt > 1 the profile is scaled by `waveform` (default aortic_waveform),
    which is quasi-steady: pressure gradient and WSS scale with it.
    """
    vtw = voxel_to_world_matrix() if voxel_to_world is None else np.asarray(voxel_to_world, float)
    r, z, n = _cylinder_geometry(shape, vtw, radius_mm, axis_world)
    mask = r < radius_mm
    profile = np.where(mask, v_max_mm_s * (1.0 - (r / radius_mm) ** 2), 0.0)
    w = _waveform(nt, waveform)
    velocity = np.stack([profile[..., None] * n * s for s in w])          # (nt, ..., 3)
    R = radius_mm * 1e-3; vmax = v_max_mm_s * 1e-3
    dpdz_pa_m = -4.0 * viscosity * vmax / R ** 2                           # Pa/m
    wss = 2.0 * viscosity * vmax / R                                       # Pa
    return dict(velocity=velocity, mask=mask, voxel_to_world=vtw, axis_world=n, radius_mm=radius_mm,
                r_mm=r, z_mm=z, waveform=w,
                pressure_gradient_pa_mm=dpdz_pa_m * 1e-3 * w, wss_pa=wss * w)


def womersley(shape=(32, 48, 48), voxel_to_world=None, radius_mm=12.0, nt=24, rr_ms=800.0,
              harmonics=((0, 60.0), (1, 1500.0), (2, 500.0)), axis_world=(0.0, 0.0, 1.0),
              viscosity=BLOOD_VISCOSITY_PA_S, density=BLOOD_DENSITY_KG_M3):
    """
    Womersley 1955 pulsatile pipe flow. `harmonics` is a list of
    (n, A_n) with the pressure gradient -dp/dz = sum_n A_n cos(n w t),
    A_n in Pa/m, w = 2 pi / RR. n = 0 is the steady Poiseuille component.

    Returns dict with velocity (nt, ..., 3), mask, voxel_to_world,
    pressure_gradient_pa_mm (nt,), wss_pa (nt,), flow_mm3_s (nt,),
    womersley_number (of the first harmonic).
    """
    vtw = voxel_to_world_matrix() if voxel_to_world is None else np.asarray(voxel_to_world, float)
    r, z, n = _cylinder_geometry(shape, vtw, radius_mm, axis_world)
    mask = r < radius_mm
    R = radius_mm * 1e-3
    omega = 2 * np.pi / (rr_ms * 1e-3)
    t = np.arange(nt) * rr_ms * 1e-3 / nt
    rr = (r * 1e-3)[None]                                           # m, (1, ...)
    v = np.zeros((nt,) + r.shape)                                   # m/s
    dpdz = np.zeros(nt)
    wss = np.zeros(nt)
    flow = np.zeros(nt)
    alpha1 = None
    for k, A in harmonics:
        if k == 0:
            v += (A / (4 * viscosity) * (R ** 2 - rr ** 2))
            dpdz += A
            wss += A * R / 2
            flow += np.pi * A * R ** 4 / (8 * viscosity)
            continue
        wn = k * omega
        alpha = R * np.sqrt(wn * density / viscosity)
        if alpha1 is None:
            alpha1 = alpha
        i32 = 1j ** 1.5
        J0R = special.jv(0, i32 * alpha)
        J0r = special.jv(0, i32 * alpha * rr / R)
        J1R = special.jv(1, i32 * alpha)
        e = np.exp(1j * wn * t)[:, None, None, None]
        # velocity (Womersley eq. 9), pressure gradient -dp/dz = A cos(wn t)
        v += np.real(A / (1j * wn * density) * (1 - J0r / J0R) * e)
        dpdz += A * np.cos(wn * t)
        # wall shear stress tau_w = -mu dv/dr at r = R (positive along +z when
        # the fluid drags the wall forward). With d/dr J0(c r) = -c J1(c r):
        # dv/dr|_R = A/(i wn rho) * (i^1.5 alpha / R) * J1R / J0R, so
        # tau_w = -mu * that.
        dvdr_R = A / (1j * wn * density) * (i32 * alpha / R) * J1R / J0R
        wss += -np.real(dvdr_R * np.exp(1j * wn * t)) * viscosity
        flow += np.real(np.pi * R ** 2 * A / (1j * wn * density) * (1 - 2 * J1R / (i32 * alpha * J0R)) * np.exp(1j * wn * t))
    v = np.where(mask[None], v, 0.0) * 1e3                           # mm/s
    velocity = v[..., None] * n
    return dict(velocity=velocity, mask=mask, voxel_to_world=vtw, axis_world=n, radius_mm=radius_mm,
                r_mm=r, z_mm=z, t_s=t, pressure_gradient_pa_mm=-dpdz * 1e-3, wss_pa=wss,
                flow_mm3_s=flow * 1e9, womersley_number=alpha1)


def stenosis(shape=(64, 40, 40), voxel_to_world=None, radius_mm=10.0, severity=0.5,
             length_mm=30.0, v_max_inlet_mm_s=800.0, axis_world=(0.0, 0.0, 1.0), nt=1,
             waveform=None, viscosity=BLOOD_VISCOSITY_PA_S, density=BLOOD_DENSITY_KG_M3):
    """
    Cosine-constricted tube: R(z) = R0 [1 - severity/2 (1 + cos(2 pi z/L))]
    for |z| < L/2, R0 elsewhere; throat radius R0 (1 - severity).

    Velocity (lubrication approximation, exactly divergence-free):
        v_z = 2 Q / (pi R^2) (1 - r^2/R^2)
        v_r = 2 Q / pi  R' r / R^3 (1 - r^2/R^2)
    with Q = pi R0^2 v_max_inlet / 2.

    Returns dict with velocity (nt, ..., 3), mask, voxel_to_world, r_mm,
    z_mm, radius_of_z (nk-length table on z), flow_mm3_s, throat: dict
    (radius_mm, mean_velocity_mm_s, peak_velocity_mm_s, z_mm),
    pressure_centreline_pa (per frame, function of z sampled on the voxel
    z grid; zero at inlet end), pressure_drop_pa (inlet to throat, per
    frame), bernoulli_drop_pa (rho/2 (v_throat^2 - v_inlet^2), per frame),
    simplified_bernoulli_mmhg (4 v_max^2 with v in m/s), wss_pa ((nt, n_z):
    2 mu v_max(z) / R(z) at the n_z distinct z values of the masked voxels,
    the same table as pressure_centreline_pa), waveform.
    """
    vtw = voxel_to_world_matrix() if voxel_to_world is None else np.asarray(voxel_to_world, float)
    r, z, n = _cylinder_geometry(shape, vtw, radius_mm, axis_world)
    R0 = float(radius_mm); L = float(length_mm); sev = float(severity)
    Q = np.pi * R0 ** 2 * v_max_inlet_mm_s / 2.0                           # mm^3/s

    def R_of(zz):
        inside = np.abs(zz) < L / 2
        return np.where(inside, R0 * (1.0 - sev / 2.0 * (1.0 + np.cos(2 * np.pi * zz / L))), R0)

    def dR_of(zz):
        inside = np.abs(zz) < L / 2
        return np.where(inside, R0 * sev / 2.0 * (2 * np.pi / L) * np.sin(2 * np.pi * zz / L), 0.0)

    R = R_of(z); dR = dR_of(z)
    mask = r < R
    with np.errstate(divide="ignore", invalid="ignore"):
        prof = np.where(mask, 1.0 - (r / R) ** 2, 0.0)
        vz = 2 * Q / (np.pi * R ** 2) * prof
        vr = 2 * Q / np.pi * dR * r / R ** 3 * prof
    # radial unit vector in world space
    pts = voxel_grid_world_points(shape, vtw)
    c = pts.reshape(-1, 3).mean(0)
    d = pts - c
    radial = d - (d @ n)[..., None] * n
    with np.errstate(divide="ignore", invalid="ignore"):
        e_r = np.where(r[..., None] > 1e-9, radial / np.maximum(r, 1e-9)[..., None], 0.0)
    v1 = vz[..., None] * n + vr[..., None] * e_r
    w = _waveform(nt, waveform)
    velocity = np.stack([v1 * s for s in w]) * mask[None, ..., None]

    # analytic centreline pressure along z (lubrication): dp/dz = -rho v_c v_c' - 8 mu Q / (pi R^4)
    zz = np.sort(np.unique(np.round(z[mask], 6)))
    Rz = R_of(zz); vc = 2 * Q / (np.pi * Rz ** 2)                          # mm/s
    Rm = Rz * 1e-3; Qm = Q * 1e-9; vcm = vc * 1e-3; zm = zz * 1e-3
    visc = -8 * viscosity * Qm / (np.pi * Rm ** 4)                          # Pa/m
    p_visc = np.concatenate([[0.0], np.cumsum(0.5 * (visc[1:] + visc[:-1]) * np.diff(zm))])
    p_inert = -0.5 * density * (vcm ** 2 - vcm[0] ** 2)
    i_throat = int(np.argmin(Rz))
    p_centre = np.stack([(p_inert + p_visc) * s ** 2 if False else p_inert * s ** 2 + p_visc * s for s in w])
    drop = p_centre[:, 0] - p_centre[:, i_throat]
    bern = 0.5 * density * (vcm[i_throat] ** 2 - vcm[0] ** 2) * w ** 2
    wss = (2 * viscosity * (vc * 1e-3) / Rm)[None, :] * w[:, None]
    return dict(velocity=velocity, mask=mask, voxel_to_world=vtw, axis_world=n, radius_mm=R0, r_mm=r, z_mm=z,
                radius_of_z=np.stack([zz, Rz], 1), flow_mm3_s=Q * w, waveform=w,
                throat=dict(radius_mm=float(Rz[i_throat]), z_mm=float(zz[i_throat]),
                            mean_velocity_mm_s=(Q / (np.pi * Rz[i_throat] ** 2) * w),
                            peak_velocity_mm_s=(vc[i_throat] * w)),
                pressure_centreline_pa=p_centre, pressure_z_mm=zz, pressure_drop_pa=drop,
                bernoulli_drop_pa=bern, simplified_bernoulli_mmhg=4.0 * (vc[i_throat] * 1e-3 * w) ** 2,
                wss_pa=wss)


def bifurcation(shape=(80, 40, 72), voxel_to_world=None, radius_mm=10.0, split=0.6, angle_deg=30.0,
                v_max_parent_mm_s=800.0, nt=1, waveform=None):
    """
    Parent along +z for z < 0 (radius R0, peak v_max_parent), daughters for
    z > 0 along (+-sin a, 0, cos a) with radii R0 split^(1/3) and
    R0 (1 - split)^(1/3) and flows split Q and (1 - split) Q.

    Returns dict with velocity (nt, ..., 3), mask, junction (bool, voxels
    within R_max / sin(a) + R0 of the origin, where the daughter cylinders
    overlap and the union field is not divergence-free), voxel_to_world,
    flow_mm3_s (parent, per frame), branches: {name: dict(axis, radius_mm,
    flow_mm3_s, v_max_mm_s)}, planes: {name: dict(centre, normal, radius)}
    in world mm, placed beyond the overlap in each branch (25 mm at the
    default 30 degrees, further for smaller angles; ValueError if the
    volume is too small for that), and the expected consistency checks.
    """
    vtw = voxel_to_world_matrix() if voxel_to_world is None else np.asarray(voxel_to_world, float)
    pts = voxel_grid_world_points(shape, vtw)
    c = pts.reshape(-1, 3).mean(0)
    d = pts - c
    R0 = float(radius_mm); f = float(split); a = np.radians(angle_deg)
    Q = np.pi * R0 ** 2 * v_max_parent_mm_s / 2.0
    branches = {
        "parent": dict(axis=np.array([0.0, 0.0, 1.0]), radius_mm=R0, flow_mm3_s=Q, side=-1),
        "daughter1": dict(axis=np.array([np.sin(a), 0.0, np.cos(a)]), radius_mm=R0 * f ** (1 / 3), flow_mm3_s=f * Q, side=+1),
        "daughter2": dict(axis=np.array([-np.sin(a), 0.0, np.cos(a)]), radius_mm=R0 * (1 - f) ** (1 / 3), flow_mm3_s=(1 - f) * Q, side=+1),
    }
    v1 = np.zeros(shape + (3,)); mask = np.zeros(shape, bool); dist_best = np.full(shape, np.inf)
    for name, b in branches.items():
        n = b["axis"]; R = b["radius_mm"]
        s_ax = d @ n
        rad = np.linalg.norm(d - s_ax[..., None] * n, axis=-1)
        region = (rad < R) & ((s_ax <= 0) if b["side"] < 0 else (s_ax >= 0))
        vmax = 2 * b["flow_mm3_s"] / (np.pi * R ** 2)
        b["v_max_mm_s"] = vmax
        prof = np.where(region, vmax * (1 - (rad / R) ** 2), 0.0)
        take = region & (rad / R < dist_best)        # nearest axis (normalised) wins in overlaps
        v1[take] = prof[take, None] * n
        dist_best[take] = (rad / R)[take]
        mask |= region
    w = _waveform(nt, waveform)
    velocity = np.stack([v1 * s for s in w])
    # The two daughter cylinders overlap along their axes out to
    # R_max / sin(a) from the origin (16.9 mm at 30 degrees, 32.6 mm at 15),
    # and the union field is not divergence-free there; the junction mask
    # and the plane distance scale with the angle so that the planes sit
    # in clean single-branch flow and a plane disc does not cut the other
    # branch. (Fixed 1.5 R0 and 25 mm were only right for angles >= 28.)
    R1, R2 = branches["daughter1"]["radius_mm"], branches["daughter2"]["radius_mm"]
    overlap = max(R1, R2) / np.sin(a)
    junction = np.linalg.norm(d, axis=-1) < overlap + R0
    # a daughter plane disc (radius R + 3) at distance s from the junction
    # reaches the other daughter (axes 2a apart) when s sin(2a) < R1 + R2 + 3
    plane_dist = max(25.0, overlap + 2.0, (R1 + R2 + 3.0) / np.sin(2 * a) + 2.0)
    lo, hi = pts.reshape(-1, 3).min(0), pts.reshape(-1, 3).max(0)
    planes = {}
    for name, b in branches.items():
        s_pl = -plane_dist if b["side"] < 0 else plane_dist
        centre = c + s_pl * b["axis"]; r_pl = float(b["radius_mm"] + 3.0)
        if np.any(centre - r_pl < lo) or np.any(centre + r_pl > hi):
            raise ValueError(f"angle {angle_deg} deg puts the {name} plane {plane_dist:.0f} mm from the junction, "
                             "outside the volume; enlarge shape or the angle")
        planes[name] = dict(centre=tuple(centre), normal=tuple(b["axis"]), radius=r_pl)
    checks = [("parent vs daughters", ["parent"], ["daughter1", "daughter2"]),
              ("daughter1 : daughter2", ["daughter1"], ["daughter2"])]
    return dict(velocity=velocity, mask=mask, junction=junction, voxel_to_world=vtw, flow_mm3_s=Q * w,
                waveform=w, branches=branches, planes=planes, checks=checks,
                expected_ratio_d1_d2=f / (1 - f), centre_world=c)


def chest(shape=(40, 96, 96), voxel_to_world=None, nt=20, rng=None, steady_vein=True, wrap=True,
          offset_coefficients=None):
    """
    Thorax phantom for background phase correction. Returns dict with
    velocity_true (nt, ..., 3) mm/s (no offset), velocity (measured, with
    offset and optional wrap band), magnitude (nt, ...), lumen (artery mask),
    static (true static-tissue mask, excluding wrap band), offset
    (nk, nj, ni, 3) mm/s, body mask, voxel_to_world.
    Units: cm/s in the offset coefficients are converted to mm/s.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    vtw = voxel_to_world_matrix() if voxel_to_world is None else np.asarray(voxel_to_world, float)
    nk, nj, ni = shape
    Z, Y, X = np.meshgrid(np.linspace(-1, 1, nk), np.linspace(-1, 1, nj), np.linspace(-1, 1, ni), indexing="ij")
    body = ((X / 0.85) ** 2 + (Y / 0.98) ** 2) < 1.0
    lungs = (((X - 0.35) ** 2 + (Y / 0.7) ** 2) < 0.22 ** 2) | (((X + 0.35) ** 2 + (Y / 0.7) ** 2) < 0.22 ** 2)
    artery = (X ** 2 + (Y + 0.2) ** 2) < 0.12 ** 2
    vein = ((X - 0.15) ** 2 + (Y + 0.45) ** 2) < 0.10 ** 2

    mag = np.where(body, 1.0, 0.05); mag[lungs] = 0.08; mag[artery] = 1.5; mag[vein] = 1.3
    magnitude = mag[None] * (1 + 0.05 * rng.normal(size=(nt,) + shape))

    pulse = (800.0 * np.clip(np.sin(2 * np.pi * np.arange(nt) / nt), 0, None) + 100.0)[:, None, None, None]
    # parabolic (Poiseuille) profile across the artery so that spatial
    # unwrapping has a gradient to work with, centre velocity = pulse
    r2 = (X ** 2 + (Y + 0.2) ** 2) / 0.12 ** 2
    profile = np.where(artery, 1.0 - r2, 0.0)
    v_true = np.zeros((nt,) + shape + (3,))
    # the artery runs along array axis k: put the velocity along that axis's
    # world direction (world z for the default geometry)
    k_dir = vtw[:3, 2] / np.linalg.norm(vtw[:3, 2])
    v_true += (pulse * profile[None])[..., None] * k_dir
    if steady_vein:
        v_true += (50.0 * vein[None])[..., None] * k_dir
    noise = rng.normal(0, 10.0, v_true.shape)
    noise[:, lungs | ~body] *= 25.0
    v_true += noise

    from .background_phase import design_matrix, normalised_coords
    A = design_matrix(normalised_coords(shape), 2)
    if offset_coefficients is None:
        offset_coefficients = 10.0 * np.array([                              # cm/s -> mm/s
            [0.8, 1.5, -1.0, 0.6, 0.0, 0.9, -0.4, 0.5, 0.3, -0.7],
            [-0.5, 0.7, 1.2, -0.9, 0.3, 0.0, 0.8, -0.6, 0.2, 0.4],
            [0.3, -1.1, 0.6, 1.3, -0.5, 0.4, 0.0, 0.7, -0.8, 0.2]])
    offset = np.stack([(A @ c).reshape(shape) for c in offset_coefficients], -1)
    v_meas = v_true + offset[None]

    static = body & ~lungs & ~(X ** 2 + (Y + 0.2) ** 2 < 0.15 ** 2) & ~((X - 0.15) ** 2 + (Y + 0.45) ** 2 < 0.13 ** 2)
    band = int(0.12 * nj)
    wrapmask = np.zeros(shape, bool)
    if wrap:
        Yf = Y.copy()
        Yf[:, :band] = -1.0 - (Y[:, :band] + 1.0) - 0.3
        Yf[:, -band:] = 1.0 + (1.0 - Y[:, -band:]) + 0.3
        Af = design_matrix(np.stack([Z.ravel(), Yf.ravel(), X.ravel()], 1), 2)
        off_fold = np.stack([(Af @ c).reshape(shape) for c in offset_coefficients], -1)
        wrapmask[:, :band] = True; wrapmask[:, -band:] = True; wrapmask &= body
        v_meas[:, wrapmask] = (noise + off_fold[None])[:, wrapmask]
        magnitude[:, wrapmask] = 0.9 * (1 + 0.05 * rng.normal(size=(nt, wrapmask.sum())))
        static &= ~wrapmask
    return dict(velocity_true=v_true, velocity=v_meas, magnitude=magnitude, lumen=artery, vein=vein,
                static=static, offset=offset, body=body, lungs=lungs, wrap_band=wrapmask, voxel_to_world=vtw)


def valve_jet(shape=(48, 48, 64), voxel_to_world=None, nt=20, orifice_radius_mm=4.0, jet_peak_mm_s=4000.0,
              jet_frames=(4, 9), spread_deg=8.0, core_length_mm=24.0, swirl_rad_s=4.0, chamber_axes_mm=(28.0, 28.0, 38.0),
              turbulence_intensity=0.2, axis_world=(0.0, 0.0, 1.0), rng=None):
    """
    Valve jet phantom: an orifice jet entering a chamber of slow swirling
    blood, as in valvar regurgitation (into an atrium or ventricle) or
    stenosis (into an outflow chamber or great artery).

    Geometry (world mm): the chamber is an ellipsoid with semi-axes
    chamber_axes_mm centred in the field of view; the orifice is a disc of
    orifice_radius_mm on the chamber wall where the jet axis (axis_world)
    enters, and the jet runs along the axis into the chamber.
    Jet: v_axial(r, z) = v_c(z) exp(-r^2 / R(z)^2) with R(z) = R0 + z tan(
    spread_deg) and v_c(z) = v_peak for z < core_length_mm, then v_peak
    core_length_mm / z (free-jet decay). Chamber: rigid-body swirl of
    angular speed swirl_rad_s about the jet axis (divergence-free), peak a
    few hundred mm/s at the chamber wall; the jet is added on top.
    Time: the jet follows a half-sine over frames jet_frames[0] to
    jet_frames[1] (inclusive start, exclusive end) and is absent otherwise;
    the swirl is constant. The field is kinematic, not a solution of the
    flow equations; it is a known answer for unwrapping tests.
    Intravoxel spread: intravoxel_std = turbulence_intensity x local jet
    speed in the shear layer (0.5 R < r < 1.5 R) and in the decaying core
    (z > core_length_mm), zero elsewhere, for the dephasing model of the
    acquisition functions.

    Returns dict: velocity (nt, nk, nj, ni, 3) mm/s, mask (chamber, bool),
    jet_core (|v_jet| > 0.5 peak at any frame), jet_edge (jet region
    outside the core, |v_jet| > 0.05 peak), chamber_only, intravoxel_std
    (nt, nk, nj, ni), orifice (centre_mm, normal, radius_mm), flow_mm3_s
    (per frame, through the orifice: pi R0^2 v_c(0) for the Gaussian
    profile), regurgitant_volume_mm3 (per cycle, rr_ms = 1000 assumed in
    the caller), waveform, voxel_to_world.
    """
    vtw = voxel_to_world_matrix() if voxel_to_world is None else np.asarray(voxel_to_world, float)
    pts = voxel_grid_world_points(shape, vtw)
    n = np.asarray(axis_world, float); n = n / np.linalg.norm(n)
    centre = pts.reshape(-1, 3).mean(0)
    a = np.asarray(chamber_axes_mm, float)
    # orthonormal frame (e1, e2, n)
    e1 = np.cross(n, [1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.cross(n, [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1); e2 = np.cross(n, e1)
    d = pts - centre
    x1, x2, z = d @ e1, d @ e2, d @ n
    chamber = (x1 / a[0]) ** 2 + (x2 / a[1]) ** 2 + (z / a[2]) ** 2 < 1.0
    z_in = z + a[2]                                        # distance from the inlet wall along the axis
    r = np.sqrt(x1 ** 2 + x2 ** 2)
    R0 = float(orifice_radius_mm)
    R = R0 + np.maximum(z_in, 0.0) * np.tan(np.radians(spread_deg))
    vc = np.where(z_in < core_length_mm, 1.0, core_length_mm / np.maximum(z_in, core_length_mm))
    profile = np.where(chamber & (z_in >= 0.0), vc * np.exp(-(r / R) ** 2), 0.0)
    jet_dir = n
    # swirl about the jet axis: v = omega n x d_perp
    d_perp = x1[..., None] * e1 + x2[..., None] * e2
    swirl = swirl_rad_s * np.cross(np.broadcast_to(n, d_perp.shape), d_perp) * 1e-3 * 1e3  # rad/s x mm = mm/s
    swirl = np.where(chamber[..., None], swirl, 0.0)
    w = np.zeros(nt)
    j0, j1 = jet_frames
    if j1 > j0:
        w[j0:j1] = np.sin(np.pi * (np.arange(j1 - j0) + 0.5) / (j1 - j0))
    velocity = np.empty((nt,) + tuple(shape) + (3,))
    sigma = np.zeros((nt,) + tuple(shape))
    shear = chamber & (z_in >= 0.0) & (r > 0.5 * R) & (r < 1.5 * R)
    decay = chamber & (z_in > core_length_mm) & (r < 1.5 * R)
    for t in range(nt):
        vj = jet_peak_mm_s * w[t] * profile
        velocity[t] = swirl + vj[..., None] * jet_dir
        sigma[t] = turbulence_intensity * np.abs(vj) * ((shear | decay) & (profile > 0.05))
    jet_any = jet_peak_mm_s * w.max() * profile
    jet_core = jet_any > 0.5 * jet_peak_mm_s
    jet_edge = (jet_any > 0.05 * jet_peak_mm_s) & ~jet_core
    flow = np.pi * R0 ** 2 * jet_peak_mm_s * w                   # mm^3/s through the orifice
    return dict(velocity=velocity, mask=chamber, jet_core=jet_core, jet_edge=jet_edge,
                chamber_only=chamber & ~(jet_core | jet_edge), intravoxel_std=sigma,
                orifice=dict(centre_mm=(centre - a[2] * n).tolist(), normal=n.tolist(), radius_mm=R0),
                flow_mm3_s=flow, regurgitant_volume_mm3=float(flow.sum() / nt), waveform=w,
                jet_peak_mm_s=float(jet_peak_mm_s), swirl_peak_mm_s=float(swirl_rad_s * max(a[0], a[1])),
                voxel_to_world=vtw)


def add_noise(velocity, sigma, mask=None, rng=None):
    """Add Gaussian noise of std `sigma` (velocity units) to each component,
    inside `mask` (broadcast over time) if given."""
    rng = np.random.default_rng(0) if rng is None else rng
    noise = rng.normal(0.0, sigma, np.shape(velocity))
    if mask is not None:
        noise = noise * np.asarray(mask, float)[..., None]
    return np.asarray(velocity, float) + noise


def alias(velocity, venc):
    """Wrap each component into [-venc, venc). Returns (wrapped, wrap_count)
    with wrapped = velocity - 2 venc wrap_count."""
    venc = np.broadcast_to(np.asarray(venc, float), (3,))
    k = np.floor((np.asarray(velocity, float) + venc) / (2 * venc))
    return velocity - 2 * venc * k, k.astype(int)


# ---------------------------------------------------------------------------
# Phase-contrast acquisition model
# ---------------------------------------------------------------------------

def encode(velocity, venc, magnitude=None, mask=None, air_magnitude=0.05):
    """
    Complex signals of a simple four-point phase-contrast acquisition.

    velocity  : (..., 3) in the units of venc.
    venc      : scalar or (3,).
    magnitude : (...) signal magnitude; default 1 inside `mask`, `air_magnitude`
                outside (or 1 everywhere without a mask).
    Returns signals (4, ...) complex: [S_0, S_x, S_y, S_z] with
    S_0 = M and S_c = M exp(i pi v_c / venc_c).
    """
    v = np.asarray(velocity, float)
    venc = np.broadcast_to(np.asarray(venc, float), (3,))
    if magnitude is None:
        magnitude = np.ones(v.shape[:-1]) if mask is None else np.where(mask, 1.0, air_magnitude)
    M = np.broadcast_to(np.asarray(magnitude, float), v.shape[:-1])
    S = np.empty((4,) + v.shape[:-1], complex)
    S[0] = M
    for c in range(3):
        S[c + 1] = M * np.exp(1j * np.pi * v[..., c] / venc[c])
    return S


def add_complex_noise(signals, snr, rng=None, reference_magnitude=1.0):
    """
    Independent complex Gaussian noise on every acquisition, std
    reference_magnitude / snr per real and imaginary channel, so that a
    voxel of magnitude `reference_magnitude` has magnitude SNR `snr`.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    sigma = float(reference_magnitude) / float(snr)
    n = rng.normal(0.0, sigma, signals.shape) + 1j * rng.normal(0.0, sigma, signals.shape)
    return signals + n


def decode(signals, venc):
    """
    Velocity (..., 3) and magnitude (...) from four-point signals:
    v_c = venc_c angle(S_c conj(S_0)) / pi, magnitude = mean |S| over the
    four acquisitions (as the scanner's magnitude image).
    """
    venc = np.broadcast_to(np.asarray(venc, float), (3,))
    phase = np.angle(signals[1:] * np.conj(signals[0]))                # (3, ...)
    v = np.moveaxis(phase, 0, -1) * (venc / np.pi)
    return v, np.abs(signals).mean(0)


def block_average(signals, factor):
    """
    Intravoxel averaging of complex signals over factor^3 blocks of the
    last three (spatial) axes; the spatial shape must be divisible by
    `factor`. Use before decode() to model partial volume at a coarser
    acquisition resolution. Works on any leading axes (acquisition, time).
    """
    s = np.asarray(signals)
    f = int(factor)
    if f == 1:
        return s
    *lead, nk, nj, ni = s.shape
    if nk % f or nj % f or ni % f:
        raise ValueError(f"spatial shape {(nk, nj, ni)} not divisible by {f}")
    s = s.reshape(*lead, nk // f, f, nj // f, f, ni // f, f)
    return s.mean(axis=(-5, -3, -1))


def dephasing_factor(intravoxel_std, venc):
    """Magnitude attenuation exp(-pi^2 sigma_v^2 / (2 venc^2)) of a voxel
    whose velocity along the encoding direction is Gaussian with standard
    deviation sigma_v (Dyverfeldt 2006). intravoxel_std (...) or (..., 3);
    returns (..., 3)."""
    sig = np.asarray(intravoxel_std, float)
    if sig.ndim == 0 or sig.shape[-1] != 3:
        sig = sig[..., None]
    venc = np.broadcast_to(np.asarray(venc, float), (3,))
    return np.exp(-0.5 * (np.pi * sig / venc) ** 2)


def _encode_dephased(v, venc, magnitude, mask, air_magnitude, intravoxel_std):
    S = encode(v, venc, magnitude, mask, air_magnitude)
    if intravoxel_std is not None:
        f = dephasing_factor(intravoxel_std, venc)
        for c in range(3):
            S[c + 1] = S[c + 1] * f[..., c]
    return S


def _sample_at_offset(v, offset_frames):
    """Linear interpolation of a (nt, ...) series at frame times shifted by
    offset_frames, periodic in time."""
    v = np.asarray(v, float)
    if offset_frames == 0 or v.ndim < 5:
        return v
    nt = v.shape[0]
    t = (np.arange(nt) + offset_frames) % nt
    i0 = np.floor(t).astype(int) % nt
    i1 = (i0 + 1) % nt
    w = (t - np.floor(t))[:, None, None, None, None]
    return (1 - w) * v[i0] + w * v[i1]


def acquire(velocity, venc, magnitude=None, mask=None, snr=20.0, downsample=1, rng=None,
            air_magnitude=0.05, return_signals=False, intravoxel_std=None):
    """
    Simulate a phase-contrast measurement of a velocity field.

    velocity   : (..., 3) true velocity (any leading time axis), units of venc.
    venc       : scalar or (3,) velocity encoding.
    magnitude  : optional signal magnitude (...); default from `mask`.
    snr        : magnitude SNR of a unit-magnitude voxel in one acquisition.
                 Velocity noise in such voxels is sqrt(2) venc / (pi snr).
    downsample : block-average the complex signals by this factor before
                 decoding (partial volume); the input should then be
                 generated at the finer resolution.
    intravoxel_std : optional velocity spread within each voxel (...) or
                 (..., 3), same units as venc: attenuates the encoded
                 signals (dephasing, stronger at low venc; see
                 dephasing_factor). The magnitude image then drops where
                 the spread is large, as in a turbulent jet.
    Returns dict: velocity (measured, aliased where |v| > venc), magnitude,
    velocity_true (the noise-free reference on the measured grid: the
    input, or for downsample > 1 the signal-weighted partial-volume
    velocity), wrap_count (wraps the measurement applied to velocity_true,
    (..., 3) int), velocity_noise_mm_s (the nominal sigma_v on the measured
    grid), and signals if requested.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    v = np.asarray(velocity, float)
    venc = np.broadcast_to(np.asarray(venc, float), (3,))
    S = _encode_dephased(v, venc, magnitude, mask, air_magnitude, intravoxel_std)
    S = add_complex_noise(S, snr, rng)
    S = block_average(S, downsample)
    v_meas, mag = decode(S, venc)
    if downsample == 1:
        v_true = v
    else:
        # The measured phase of a partial-volume voxel is that of the
        # signal-weighted complex average, not of the plain average of the
        # velocities (a voxel half in a 900 mm/s jet at venc 500 measures the
        # wrapped 814 mm/s of the signal-weighted average, not 450). The
        # reference and its wrap count are therefore decoded from the noise-
        # free block-averaged signals at a venc large enough not to wrap.
        venc_big = np.broadcast_to(4.0 * np.abs(v).max() + venc, (3,))
        S_clean = block_average(encode(v, venc_big, magnitude, mask, air_magnitude), downsample)
        v_true, _ = decode(S_clean, venc_big)
    _, k = alias(v_true, venc)
    # complex block averaging over factor^3 voxels lowers the velocity noise
    # by factor^-1.5 (sigma_v scales with 1/magnitude, which grows by
    # factor^3, times sqrt(factor^3) for the averaged noise)
    out = {"velocity": v_meas, "magnitude": mag, "wrap_count": k, "velocity_true": v_true,
           "velocity_noise_mm_s": (np.sqrt(2.0) * venc / (np.pi * snr) * downsample ** -1.5).tolist()}
    if return_signals:
        out["signals"] = S
    return out


def acquire_multivenc(velocity, vencs, magnitude=None, mask=None, snr=20.0, downsample=1, rng=None,
                      air_magnitude=0.05, shared_reference=True, intravoxel_std=None, time_offset_frames=None):
    """
    Simulate a multi-venc phase-contrast acquisition of one velocity field.

    vencs            : list of scalar or (3,) vencs, one per acquisition.
    shared_reference : True for an interleaved acquisition with one reference
                       scan common to all vencs (its noise is shared, so the
                       velocity noise of the encodings is correlated, as in
                       Franco 2022's variance analysis); False for separate
                       single-venc scans with independent noise.
    intravoxel_std   : optional intravoxel velocity spread (dephasing,
                       stronger at low venc; see acquire()).
    time_offset_frames : optional list, one per venc, of frame-time offsets
                       (fractions of a frame) at which that encoding samples
                       the field (linear interpolation, periodic); models the
                       timing differences between interleaved encodings.
    Other arguments as acquire().

    Returns a list of acquire()-style dicts, one per venc, in the order
    given; velocity_true is the field at the first encoding's timing and
    'magnitude' is that encoding's magnitude image.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    v0 = np.asarray(velocity, float)
    offsets = [0.0] * len(vencs) if time_offset_frames is None else list(time_offset_frames)
    if len(offsets) != len(vencs):
        raise ValueError("time_offset_frames needs one entry per venc")
    ref_noise = None
    out = []
    for venc, off in zip(vencs, offsets):
        venc = np.broadcast_to(np.asarray(venc, float), (3,))
        v = _sample_at_offset(v0, off)
        S = _encode_dephased(v, venc, magnitude, mask, air_magnitude, intravoxel_std)
        if shared_reference:
            if ref_noise is None:
                ref_noise = add_complex_noise(np.zeros_like(S[:1]), snr, rng)[0]
            S[1:] = add_complex_noise(S[1:], snr, rng)
            S[0] = S[0] + ref_noise
        else:
            S = add_complex_noise(S, snr, rng)
        S = block_average(S, downsample)
        v_meas, mag = decode(S, venc)
        if downsample == 1:
            v_true = v0
        else:
            venc_big = np.broadcast_to(4.0 * np.abs(v0).max() + venc, (3,))
            S_clean = block_average(encode(v0, venc_big, magnitude, mask, air_magnitude), downsample)
            v_true, _ = decode(S_clean, venc_big)
        _, k = alias(v_true, venc)
        out.append({"velocity": v_meas, "magnitude": mag, "wrap_count": k, "velocity_true": v_true,
                    "venc": venc.tolist(), "time_offset_frames": float(off),
                    "velocity_noise_mm_s": (np.sqrt(2.0) * venc / (np.pi * snr) * downsample ** -1.5).tolist()})
    return out
