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
               Cambridge University Press, 1967, section 4.2 (Poiseuille
               flow). ISBN 9780521663960.

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
               PMID 637964 (simplified Bernoulli).

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
acquire()      Phase-contrast forward model. Each velocity component is the
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
               (partial volume) at the vessel wall. This is the model
               behind the noise, venc and resolution considerations of the
               4D flow consensus statement.
               References:
               Pelc NJ, Herfkens RJ, Shimakawa A, Enzmann DR. Phase
               contrast cine magnetic resonance imaging. Magn Reson Q
               1991;7(4):229-254. PMID 1790111. (Velocity noise relation.)
               Pelc NJ, Bernstein MA, Shimakawa A, Glover GH. Encoding
               strategies for three-direction phase-contrast MR imaging of
               flow. J Magn Reson Imaging 1991;1(4):405-413. PMID 1790362.
               doi:10.1002/jmri.1880010404. (Four-point encoding.)
               Bernstein MA, King KF, Zhou XJ. Handbook of MRI Pulse
               Sequences. Elsevier Academic Press, 2004, chapter 15.2
               (phase contrast). ISBN 9780120928613.
               Dyverfeldt P, Bissell M, Barker AJ, et al. 4D flow
               cardiovascular magnetic resonance consensus statement.
               J Cardiovasc Magn Reson 2015;17:72. PMID 26257141.
               doi:10.1186/s12968-015-0174-5.
block_average() Intravoxel averaging of complex signals (partial volume) to
               a coarser grid; the correct way to downsample phase-contrast
               data, as opposed to averaging velocities.

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

BLOOD_VISCOSITY_PA_S = 3.5e-3      # Pa s, typical for blood at high shear
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
    w = np.ones(1) if nt == 1 else (aortic_waveform(nt) if waveform is None else np.asarray(waveform, float))
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
    simplified_bernoulli_mmhg (4 v_max^2 with v in m/s), wss_pa (nk x nt:
    2 mu v_max(z) / R(z)), waveform.
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
    w = np.ones(1) if nt == 1 else (aortic_waveform(nt) if waveform is None else np.asarray(waveform, float))
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
    v_true[..., 2] = pulse * profile[None]                                  # along world z (array axis k)
    if steady_vein:
        v_true[..., 2] += 50.0 * vein[None]
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


def acquire(velocity, venc, magnitude=None, mask=None, snr=20.0, downsample=1, rng=None,
            air_magnitude=0.05, return_signals=False):
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
    Returns dict: velocity (measured, aliased where |v| > venc), magnitude,
    wrap_count (true wraps removed by the measurement, (..., 3) int),
    velocity_noise_mm_s (the nominal sigma_v), and signals if requested.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    v = np.asarray(velocity, float)
    venc = np.broadcast_to(np.asarray(venc, float), (3,))
    S = encode(v, venc, magnitude, mask, air_magnitude)
    S = add_complex_noise(S, snr, rng)
    S = block_average(S, downsample)
    v_meas, mag = decode(S, venc)
    # true wrap count of the (block-averaged) true velocity
    v_true = v if downsample == 1 else np.moveaxis(
        np.stack([block_average(v[..., c], downsample) for c in range(3)]), 0, -1)
    _, k = alias(v_true, venc)
    out = {"velocity": v_meas, "magnitude": mag, "wrap_count": k, "velocity_true": v_true,
           "velocity_noise_mm_s": (np.sqrt(2.0) * venc / (np.pi * snr)).tolist()}
    if return_signals:
        out["signals"] = S
    return out
