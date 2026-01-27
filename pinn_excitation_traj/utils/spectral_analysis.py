import numpy as np

def spectral_analysis(
    signal,
    *,
    dt=None,
    t=None,
    remove_mean=False,
):
    """
    Frequency spectrum analysis for robot trajectories.

    Parameters
    ----------
    signal : ndarray, shape (T, dof)
        q / dq / ddq trajectory
    dt : float, optional
        Sampling period [s]
    t : ndarray, optional
        Time array [s], shape (T,)
    remove_mean : bool
        Remove DC component

    Returns
    -------
    freq : ndarray, shape (F,)
        Frequency axis (Hz)
    mag : ndarray, shape (F, dof)
        Magnitude spectrum
    energy_cum : ndarray, shape (F, dof)
        Cumulative normalized spectral energy
    """

    signal = np.asarray(signal)
    assert signal.ndim == 2, "signal must be (T, dof)"

    T, dof = signal.shape

    # --- sampling info ---
    if dt is None:
        assert t is not None, "Either dt or t must be provided"
        dt = t[1] - t[0]

    fs = 1.0 / dt  # sampling frequency

    # --- preprocessing ---
    x = signal.copy()
    if remove_mean:
        x -= np.mean(x, axis=0, keepdims=True)

    # --- FFT ---
    fft_val = np.fft.rfft(x, axis=0)
    mag = np.abs(fft_val) / T
    freq = np.fft.rfftfreq(T, d=dt)

    # --- spectral energy ---
    energy = mag ** 2
    energy_cum = np.cumsum(energy, axis=0)
    energy_cum /= energy_cum[-1, :]  # normalize to [0, 1]

    return freq, mag, energy_cum


def single_fft(signal, Fs):
    """
    计算信号的单边幅值谱和相位谱，等效于你给的 MATLAB 代码。

    参数：
    - signal: 1D numpy array，时域信号
    - Fs: 采样频率（Hz）

    返回：
    - f: 频率向量（Hz）
    - P1: 单边幅值谱
    - theta1: 单边相位谱（弧度）
    """

    # L = len(signal)
    # Y = np.fft.fft(signal)

    # P2 = np.abs(Y) / L  # 双边幅值谱
    # P1 = P2[:L//2 + 1]  # 单边幅值谱
    # P1[1:-1] = 2 * P1[1:-1]  # 乘以2，除了第一个和最后一个元素

    # theta2 = np.angle(Y)  # 双边相位谱
    # theta1 = theta2[:L//2 + 1]  # 单边相位谱

    # f = Fs * np.arange(0, L//2 + 1) / L  # 频率向量

    # return f, P1, theta1

    signal = np.asarray(signal)
    if signal.ndim == 1:
        signal = signal[:, None]  # 转成 (T, 1)

    L, dof = signal.shape
    Y = np.fft.fft(signal, axis=0)  # 对每列FFT，返回 (L, dof)

    P2 = np.abs(Y) / L               # 双边幅值谱，(L, dof)
    P1 = P2[:L//2 + 1, :]           # 单边幅值谱，(F, dof)
    if L > 1:
        P1[1:-1, :] *= 2            # 乘以2，除了第一个和最后一个元素

    theta2 = np.angle(Y)             # 双边相位谱，(L, dof)
    theta1 = theta2[:L//2 + 1, :]   # 单边相位谱，(F, dof)

    f = Fs * np.arange(0, L//2 + 1) / L  # 频率向量，(F,)

    # 如果原始输入是一维，返回对应一维结果
    if P1.shape[1] == 1:
        return f, P1[:, 0], theta1[:, 0]
    else:
        return f, P1, theta1