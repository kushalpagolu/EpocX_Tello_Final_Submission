import numpy as np
from scipy.signal import butter, filtfilt
import logging

logger = logging.getLogger(__name__)

def bandpass_filter(data, fs, lowcut=1.0, highcut=50.0, order=4):
    """
    Apply a bandpass filter to the data.
    """
    nyquist = 0.5 * fs
    low, high = lowcut / nyquist, highcut / nyquist
    b, a = butter(order, [low, high], btype='band', analog=False)
    filtered_data = filtfilt(b, a, data)
    logger.info(f"Filtered data shape: {filtered_data.shape}")
    return filtered_data

def apply_hanning_window(data):
    """
    Apply a Hanning window to the data.
    """
    window = np.hanning(len(data)) * 2
    windowed_data = (data.T * window).T
    logger.info(f"Windowed data shape: {windowed_data.shape}")
    return windowed_data
