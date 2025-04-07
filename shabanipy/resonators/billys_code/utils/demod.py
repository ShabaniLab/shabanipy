import numpy as np
import scipy


def integrate_averaged_segments_cos(t, v, f):
    return _integrate_averaged_segments(np.cos, t, v, f)


def integrate_averaged_segments_sin(t, v, f):
    return _integrate_averaged_segments(np.sin, t, v, f)


def _integrate_averaged_segments(func, t, v, f):
    """
    perform the integral of cos(2*pi*f*tau) * v(tau) dtau for segments in time
    of length 1/f
    See Schuster thesis for detail around page 149:
    http://schusterlab.uchicago.edu/static/pdfs/Schuster_thesis.pdf

    integrates (1 / f / dt) segments
    """
    N = len(t)
    dtau = t[1] - t[0]
    points_per_segment = int(1 / f / dtau)
    N_segments = N // points_per_segment
    s = np.zeros(N_segments)
    for n in range(0, N_segments):
        for m in range(n * points_per_segment, (n + 1) * points_per_segment + 1):
            s[n] += func(2 * np.pi * f * t[m]) * v[m] * dtau
    return s


def _integrate_averaged_segments_more_points(func, t, v, f):
    """
    perform the integral of cos(2*pi*f*tau) * v(tau) dtau for segments in time
    of length 1/f
    See Schuster thesis for detail around page 149:
    http://schusterlab.uchicago.edu/static/pdfs/Schuster_thesis.pdf

    integrates one value every N point segment by 0 to N, 1 to N + 1, 2 to N + 2, ...
    """
    N = len(t)
    dtau = t[1] - t[0]
    points_per_segment = int(1 / f / dtau)
    N_segments = N - points_per_segment
    s = np.zeros(N_segments)
    for n in range(0, N_segments):
        for m in range(n, n + points_per_segment):
            s[n] += func(2 * np.pi * f * t[m]) * v[m] * dtau
    return s


def demodulate_IF(t_data_sec, if_data_volt, if_freq_hz):
    """
    Single channel digital homodyne demodulation
    limited to bandwidth of if_freq_hz

    IF(t) = A(t) sin[f*t + phase(t)]

    extract A(t) and phase(t) by integrating over periods for if_freq

    See Schuster thesis for detail around page 149: http://schusterlab.uchicago.edu/static/pdfs/Schuster_thesis.pdf
    """
    t = t_data_sec
    v = if_data_volt
    f = if_freq_hz
    I = f * integrate_averaged_segments_cos(t, v, f)
    Q = f * integrate_averaged_segments_sin(t, v, f)
    A = np.sqrt(I * I + Q * Q)
    P = np.arctan(Q / I)
    return A, P


def demodulate_IQ(T, I, Q, if_freq):
    R = np.array(
        [
            [np.cos(if_freq * T), np.sin(if_freq * T)],
            [-np.sin(if_freq * T), np.cos(if_freq * T)],
        ]
    )
    r_I = np.zeros(len(I))
    r_Q = np.zeros(len(Q))
    j = 0
    for t, i, q in zip(T, I, Q):
        ri = np.cos(if_freq * t) * i + np.sin(if_freq * t) * q
        rq = -np.sin(if_freq * t) * i + np.cos(if_freq * t) * q
        r_I[j] = ri
        r_Q[j] = rq
        j += 1
    # IQ = np.array([[I], [Q]])
    # print(R.shape, IQ.shape)
    # rotated_data = np.matmul(
    #     R,
    # )
    # I = rotated_data[0]
    # Q = rotated_data[1]
    return np.sqrt(r_I * r_I + r_Q * r_Q), np.arctan(r_Q / r_I)
