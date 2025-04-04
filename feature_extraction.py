import numpy as np
from scipy.stats import entropy
from scipy.signal import welch, butter, filtfilt
from numpy.fft import fft
import logging
from scipy.signal import iirnotch
import pywt
from sklearn.preprocessing import StandardScaler
from LMSFilter import LMSFilter
from sklearn.decomposition import FastICA
from sklearn.exceptions import ConvergenceWarning  # Import ConvergenceWarning
logger = logging.getLogger(__name__)



def apply_ica(eeg_data, n_components=None):
    """
    Apply Independent Component Analysis (ICA) to remove artifacts from EEG data.
    Ensures that output retains the original time dimension.
    """
    num_channels, num_samples = eeg_data.shape  # (14, 256)
    if n_components is None:
        n_components = num_channels  # Keep the same number of components

    ica = FastICA(n_components=n_components, random_state=42, max_iter=1000)
    sources = ica.fit_transform(eeg_data.T)  # Transpose: (256, 14) → ICA → (256, 14)
    cleaned_data = ica.inverse_transform(sources).T  # Transform back & transpose to (14, 256)

    return cleaned_data  # Ensure shape remains (14, 256)



def apply_notch_filter(eeg_data, fs=256, freq=50.0, q=30.0):
    b, a = iirnotch(freq, q, fs)
    return filtfilt(b, a, eeg_data, axis=0)

def common_average_reference(eeg_data):
    """
    Apply common average reference (CAR) to EEG data.
    Handles both 1D and 2D input arrays.
    """
    if eeg_data.ndim == 1:
        mean_signal = np.mean(eeg_data, keepdims=True)
    else:
        mean_signal = np.mean(eeg_data, axis=1, keepdims=True)
    return eeg_data - mean_signal


def normalize_features(feature_matrix):
    scaler = StandardScaler()
    return scaler.fit_transform(feature_matrix)


def apply_anc(eeg_data, noise_ref, mu=0.01, n=4):
    anc_filter = LMSFilter(n=n, mu=mu)
    return eeg_data - anc_filter.predict(noise_ref)


def higuchi_fractal_dimension(eeg_signal, kmax=10):
    n = len(eeg_signal)
    lk = []
    x = np.arange(n)

    for k in range(1, kmax + 1):
        lm = np.zeros((k,))
        for m in range(k):
            indices = np.arange(m, n, k)
            lm[m] = np.sum(np.abs(np.diff(eeg_signal[indices]))) / (len(indices) * k)
        lk.append(np.mean(lm))

    coeffs = np.polyfit(np.log(np.arange(1, kmax + 1)), np.log(lk), 1)
    return -coeffs[0]  # Fractal dimension


def apply_dwt_denoising(eeg_data, wavelet='db4', level=1):
    coeffs = pywt.wavedec(eeg_data, wavelet, level=level, axis=0)
    coeffs[-1] = np.zeros_like(coeffs[-1])  # Remove high-frequency noise
    return pywt.waverec(coeffs, wavelet, axis=0)




def apply_bandpass_filter(signal, lowcut=1.0, highcut=50.0, sampling_rate=256):
    """
    Apply a bandpass filter to the given signal.
    """
    from scipy.signal import butter, filtfilt

    # Design the bandpass filter
    nyquist = 0.5 * sampling_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(4, [low, high], btype='band')

    # Apply the filter
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def apply_hanning_window(data):
    """
    Apply a Hanning window to the data.
    """
    window = np.hanning(len(data)) * 2  # Multiply by 2 as per Emotiv's method
    return (data.T * window).T

def compute_band_power(eeg_data, buffer_size=None, fs=256):
    """
    Compute band power for different frequency bands.

    Parameters
    ----------
    eeg_data : ndarray
        The EEG data to process, with shape (samples, channels).
    buffer_size : int, optional
        The size of the buffer used for FFT. If None, it defaults to the number of samples in eeg_data.
    fs : int, optional
        The sampling frequency of the EEG data. Default is 256.

    Returns
    -------
    band_powers_db : dict
        Band powers in dB for each frequency band.
    """
    if buffer_size is None:
        buffer_size = eeg_data.shape[0]  # Use the number of samples as the buffer size

    band_ranges = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 12),
        'beta': (12, 30),
        'gamma': (30, 45)
    }

    hanning_window = np.hanning(buffer_size) * 2
    eeg_data = np.array(eeg_data)
    eeg_norm = eeg_data - np.mean(eeg_data, axis=0)  # Remove DC component

    # Ensure the Hanning window matches the number of samples
    if eeg_norm.shape[0] != hanning_window.shape[0]:
        hanning_window = np.hanning(eeg_norm.shape[0]) * 2

    eeg_fft = (eeg_norm.T * hanning_window).T  # Apply Hanning window

    # Compute FFT and normalize
    fourier_transform = fft(eeg_fft, axis=0) / eeg_fft.shape[0]
    fourier_transform_norm = (2 * np.abs(fourier_transform)) / hanning_window.shape[0]
    eeg_fft_square = fourier_transform_norm ** 2

    band_powers = {}
    for band, (low, high) in band_ranges.items():
        band_idx = np.logical_and(low <= np.fft.fftfreq(eeg_norm.shape[0], 1 / fs),
                                  np.fft.fftfreq(eeg_norm.shape[0], 1 / fs) <= high)
        band_power = np.sum(eeg_fft_square[band_idx]) / len(band_idx)
        band_powers[band] = band_power if band_power > 0 else 1e-10  # Avoid log(0) errors
        logger.info(f"Band Power for {band}: {band_powers[band]}")

    # Convert to dB scale
    band_powers_db = {band: 10 * np.log10(power) for band, power in band_powers.items()}
    logger.info(f"Computed Band Powers Length: {len(band_powers_db)}")  # Log the length of the dictionary

    return band_powers_db

def compute_hjorth_parameters(eeg_signal):
    """
    Compute Hjorth parameters for EEG signal.
    """
    first_derivative = np.diff(eeg_signal)
    second_derivative = np.diff(first_derivative)
    var_zero = np.var(eeg_signal)
    var_d1 = np.var(first_derivative)
    var_d2 = np.var(second_derivative)
    mobility = np.sqrt(var_d1 / var_zero) if var_zero != 0 else 0.0
    complexity = np.sqrt(var_d2 / var_d1) / mobility if var_d1 != 0 else 0.0
    logger.info(f"Hjorth parameters: mobility={mobility}, complexity={complexity}")
    return np.array([mobility, complexity])

def compute_spectral_entropy(eeg_signal, fs):
    """
    Compute spectral entropy for EEG signal.
    """
    freqs, psd = welch(eeg_signal, fs, nperseg=min(len(eeg_signal), fs))
    psd_sum = np.sum(psd)
    if (psd_sum == 0):
        logger.warning("Spectral entropy calculation skipped due to zero PSD sum.")
        return 0
    psd_norm = psd / psd_sum
    entropy_value = entropy(psd_norm)
    logger.info(f"Spectral entropy: {entropy_value}, PSD shape: {psd.shape}")
    return entropy_value
