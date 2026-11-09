import numpy as np

from spaceops import sensing as se
from spaceops.constants import C, SIGMA_SB


def rel(a, b):
    return abs(a - b) / max(abs(a), abs(b))


def test_radar_range_closed_form_inverts_snr():
    kw = dict(p_tx=1e6, gain=10**4.0, wavelength=0.03, rcs=100.0)
    for snr in (1.0, 13.0, 100.0):
        R = se.radar_range_for_snr(snr=snr, **kw)
        assert rel(se.radar_snr(r=R, **kw), snr) < 1e-12


def test_radar_range_scales_as_fourth_root_of_power_and_rcs():
    base = dict(gain=10**4.0, wavelength=0.03, snr=13.0)
    r1 = se.radar_range_for_snr(p_tx=1e6, rcs=10.0, **base)
    assert rel(se.radar_range_for_snr(p_tx=16e6, rcs=10.0, **base) / r1, 2.0) < 1e-12
    assert rel(se.radar_range_for_snr(p_tx=1e6, rcs=160.0, **base) / r1, 2.0) < 1e-12
    # R^-4 law in SNR
    kw = dict(p_tx=1e6, gain=10**4.0, wavelength=0.03, rcs=10.0)
    assert rel(se.radar_snr(r=1e6, **kw) / se.radar_snr(r=2e6, **kw), 16.0) < 1e-12


def test_radar_range_is_known_value():
    # hand calc: R = (Pt G^2 lam^2 sigma / ((4pi)^3 SNR k T B F L))^(1/4)
    Pt, G, lam, s = 1e6, 1e4, 0.03, 100.0
    k, T, B, F, L, snr = 1.380649e-23, 500.0, 1e6, 2.0, 2.0, 13.0
    expect = (Pt * G**2 * lam**2 * s / ((4 * np.pi) ** 3 * snr * k * T * B * F * L)) ** 0.25
    got = se.radar_range_for_snr(Pt, G, lam, s, snr)
    assert rel(got, expect) < 1e-6


def test_passive_range_scales_as_square_root_and_beats_radar_scaling():
    nei = 1e-14
    I = se.ir_intensity(300.0, 100.0)
    r1 = se.passive_range_for_snr(I, 10.0, nei)
    r2 = se.passive_range_for_snr(4 * I, 10.0, nei)
    assert rel(r2 / r1, 2.0) < 1e-12  # R^2 law: 4x intensity, 2x range
    assert rel(se.passive_snr(I, r1, nei), 10.0) < 1e-12
    assert rel(se.passive_snr(I, r1, nei) / se.passive_snr(I, 2 * r1, nei), 4.0) < 1e-12


def test_ir_intensity_stefan_boltzmann_and_drive_plume():
    assert rel(se.ir_intensity(300.0, 10.0, 1.0), SIGMA_SB * 300.0**4 * 10.0 / np.pi) < 1e-12
    # a 1 GW plume (8e7 W/sr) is hundreds of times brighter than a 300 K, 1000 m^2 hull (1.3e5 W/sr)
    ratio = se.plume_intensity(1e9) / se.ir_intensity(300.0, 1000.0)
    assert 400.0 < ratio < 800.0, ratio


def test_bright_background_masks_target():
    I = se.ir_intensity(300.0, 100.0)
    nei = 1e-14
    clear = se.passive_snr(I, 1e6, nei, background_radiance=1e-6)
    masked = se.passive_snr(I, 1e6, nei, background_radiance=150.0)  # ~280 K planet disk
    assert masked < 0.01 * clear


def test_detection_probability_monotonic():
    ps = [se.detection_probability(s, 5.0) for s in (0.0, 3.0, 5.0, 7.0, 9.0)]
    assert all(b > a for a, b in zip(ps, ps[1:]))
    assert abs(ps[2] - 0.5) < 1e-12


def test_light_delay_stationary_is_range_over_c():
    pos = np.array([3.0e8, 0.0, 0.0])
    d = se.light_delay(np.zeros(3), 100.0, lambda t: pos)
    assert rel(d, 3.0e8 / C) < 1e-12


def test_light_delay_receding_target_closed_form():
    R0, v = 3.0e8, 2.0e5  # at t_obs = 0 the target is at R0 receding at v
    fn = lambda t: np.array([R0 + v * t, 0.0, 0.0])
    d = se.light_delay(np.zeros(3), 0.0, fn)
    assert rel(d, R0 / (C + v)) < 1e-12


def test_light_delay_approaching_target_closed_form():
    R0, v = 3.0e8, 2.0e5
    fn = lambda t: np.array([R0 - v * t, 0.0, 0.0])
    d = se.light_delay(np.zeros(3), 0.0, fn)
    assert rel(d, R0 / (C - v)) < 1e-12


def test_light_delay_across_the_solar_system_is_minutes():
    # 1 AU is about 499 s
    d = se.light_delay(np.zeros(3), 0.0, lambda t: np.array([1.495978707e11, 0, 0]))
    assert abs(d - 499.0) < 0.1


def test_measurement_covariance_axes():
    obs = np.zeros(3)
    tgt = np.array([1.0e6, 0.0, 0.0])
    R = se.measurement_covariance(obs, tgt, sigma_angle=1e-4, sigma_range=10.0)
    assert rel(R[0, 0], 100.0) < 1e-12  # along LOS
    assert rel(R[1, 1], (1e-4 * 1e6) ** 2) < 1e-12  # across
    assert np.allclose(R, R.T)


def _run_track(n=60, dt=5.0, seed=3, sigma_angle=2e-5, sigma_range=30.0):
    rng = np.random.default_rng(seed)
    obs = np.zeros(3)
    p0 = np.array([4.0e6, 1.0e6, -5.0e5])
    v = np.array([-800.0, 150.0, 40.0])
    trk = None
    chi = []
    for k in range(n):
        t = k * dt
        truth = p0 + v * t
        R = se.measurement_covariance(obs, truth, sigma_angle, sigma_range)
        z = truth + rng.multivariate_normal(np.zeros(3), R)
        if trk is None:
            trk = se.Track.initiate(z, R, t, q=0.01)
        else:
            trk.predict(t)
            chi.append(trk.mahalanobis_sq(truth))
            trk.update(z, R, t)
    return trk, truth, v, chi


def test_filter_converges_to_truth_and_shrinks_ellipsoid():
    trk, truth, v, _ = _run_track()
    assert np.linalg.norm(trk.position() - truth) < 400.0
    assert np.linalg.norm(trk.velocity() - v) < 5.0
    first = se.Track.initiate(truth, se.measurement_covariance(np.zeros(3), truth, 2e-5, 30.0), 0.0)
    assert trk.ellipsoid_semi_axes()[0] < first.ellipsoid_semi_axes()[0]


def test_filter_is_consistent_chi_square_of_predicted_error():
    # mean Mahalanobis^2 of the *predicted* error should be about 3 (dof) for a consistent filter
    means = []
    for seed in range(20):
        _, _, _, chi = _run_track(seed=seed)
        means.append(np.mean(chi[10:]))
    m = float(np.mean(means))
    assert 2.0 < m < 4.5, m


def test_track_quality_progression():
    obs = np.zeros(3)
    truth = np.array([4.0e6, 0.0, 0.0])
    R_bad = se.measurement_covariance(obs, truth, 1e-3, 5000.0)
    trk = se.Track.initiate(truth, R_bad, 0.0)
    assert se.classify_track(None) == se.TrackQuality.NONE
    assert se.classify_track(trk) == se.TrackQuality.DETECTION  # too few updates
    R_good = se.measurement_covariance(obs, truth, 1e-6, 5.0)
    for k in range(1, 40):
        trk.update(truth, R_good, 2.0 * k)
    assert se.classify_track(trk) == se.TrackQuality.FIRE_CONTROL
    # a coarse sensor never reaches fire control
    R_coarse = se.measurement_covariance(obs, truth, 1e-2, 5.0e4)
    coarse = se.Track.initiate(truth, R_coarse, 0.0)
    for k in range(1, 10):
        coarse.update(truth, R_coarse, 2.0 * k)
    assert se.classify_track(coarse) < se.TrackQuality.FIRE_CONTROL


def test_stale_track_ellipsoid_grows_without_updates():
    obs = np.zeros(3)
    truth = np.array([4.0e6, 0.0, 0.0])
    R = se.measurement_covariance(obs, truth, 1e-5, 20.0)
    trk = se.Track.initiate(truth, R, 0.0)
    for k in range(1, 30):
        trk.update(truth, R, 2.0 * k)
    a0 = trk.ellipsoid_semi_axes()[0]
    trk.predict(trk.t + 600.0)
    assert trk.ellipsoid_semi_axes()[0] > 10 * a0
