import hid
import numpy as np
from Crypto.Cipher import AES
from datetime import datetime
import logging
import os
import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from collections import deque
from scipy.signal import butter, filtfilt, welch
from scipy.stats import entropy
from numpy.fft import fft
from kalman_filter import KalmanFilter  # Ensure the correct import
from feature_extraction import (
    apply_bandpass_filter, apply_notch_filter, common_average_reference,
    apply_ica, apply_anc, apply_hanning_window, apply_dwt_denoising,
    compute_band_power, compute_hjorth_parameters, compute_spectral_entropy,
    higuchi_fractal_dimension, normalize_features
)
from filtering import bandpass_filter
# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
# Ensure the logs directory exists
logs_dir = "/Users/kushalpagolu/Desktop/EmotivEpoch/epoch_tello_RL_3DBrain/logs"
os.makedirs(logs_dir, exist_ok=True)

# Generate a timestamped log file name
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f"{logs_dir}/stream_data_{timestamp}.log"

# Update logging configuration to save logs to a new file for each run
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),  # Save logs to a timestamped file
        logging.StreamHandler()  # Also display logs in the console
    ]
)

class EmotivStreamer:
    logger = logging.getLogger(__name__)

    def __init__(self):
        self.vid = 0x1234
        self.pid = 0xed02
        self.device = None
        self.cipher = None
        self.cypher_key = bytes.fromhex("31003554381037423100354838003750")
        self.channel_names = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
        self.fs = 256  # Sampling frequency
        self.buffer_size = self.fs  # 1-second buffer size
        self.primary_buffer = self.initialize_buffers(self.channel_names, self.buffer_size)
        self.secondary_buffer = self.initialize_buffers(self.channel_names, self.buffer_size)
        self.processing_in_progress = False  # Flag to indicate if processing is happening
        self.logger = logging.getLogger(__name__)
        self.consecutive_invalid_packets = 0  # Counter for invalid packets
        self.max_invalid_packets = 45  # Threshold for reconnection
        self.is_buffer_ready = False  # Initialize buffer readiness flag
        self.eeg_buffers = {channel: [] for channel in self.channel_names}
        self.buffer_size = 256  # Example buffer size
        self.required_buffer_size = self.fs  # Set required buffer size to match 1 second of data (256 samples)

        self.reference_channel = 0  # Example reference channel index
        self.use_ica = True
        self.use_dwt = True
        self.use_hfd = True
        self.use_bandpass = True
        self.use_hjorth = True
        self.use_entropy = True
        self.use_bandpower = True
        self.use_hanning = True
        self.use_anc = False
        self.feature_window = deque(maxlen=10)  # Store the last 10 feature vectors (10 seconds)


    def initialize_buffers(self, channel_names, buffer_size):
        # Configure logging
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        """
        Initialize rolling buffers for EEG data.
        """
        buffers = {ch: deque(maxlen=buffer_size) for ch in channel_names}
        self.logger.info(f"Initialized buffers for channels: {channel_names}")
        return buffers




    def connect(self):
            try:
                self.device = hid.device()

                self.device.open(self.vid, self.pid)

                if self.device is None:
                    self.logger.error("Device object is None after opening. Check VID/PID or permissions.")
                    return False

                self.logger.info(f"Connected to Emotiv device {self.vid:04x}:{self.pid:04x}")
                self.device.set_nonblocking(1)
                self.cipher = AES.new(self.cypher_key, AES.MODE_ECB)
                return True

            except Exception as e:
                self.logger.error(f"Connection failed: {str(e)}")
                return False

    def disconnect(self):
        if self.device:
            self.device.close()
            self.logger.info("Disconnected from Emotiv device")


    def read_emotiv_data(self):
        """
        Reads a packet from the Emotiv device and handles empty packets gracefully.
        """
        try:
            encrypted = bytes(self.device.read(32))
            #self.logger.info(f"Encrypted packet length: {len(encrypted)}")

            if len(encrypted) == 0:
               # self.logger.warning("Empty packet received. Skipping...")
                return None

            if len(encrypted) != 32:
               # self.logger.error(f"Invalid packet length ({len(encrypted)}). Skipping packet.")
                return None

            decrypted = self.cipher.decrypt(encrypted)
            #self.logger.info(f"Decrypted packet length: {len(decrypted)}")

            packet = list(decrypted)
            packet_dict = {
                'timestamp': datetime.now().isoformat(),
                'counter': decrypted[0],
                'gyro_x': decrypted[29],
                'gyro_y': decrypted[30],
                'battery': (decrypted[31] & 0x0F)
            }

            for i, channel_name in enumerate(self.channel_names):
                start_idx = 2 * i + 1
                end_idx = start_idx + 2
                packet_dict[channel_name] = int.from_bytes(decrypted[start_idx:end_idx], 'big', signed=True)

            #self.logger.info(f"Packet received")
            return packet_dict

        except Exception as e:
            self.logger.error(f"Error reading packet: {e}")
            return None

    def update_eeg_buffers(self, raw_data, channel_names, primary_buffer, secondary_buffer, processing_in_progress):
        """
        Updates the EEG buffers with new raw data and processes features.
        Args:
            raw_data (dict): Raw EEG data packet.
            channel_names (list): List of EEG channel names.
            primary_buffer (dict): Primary buffer for EEG data.
            secondary_buffer (dict): Secondary buffer for EEG data.
            processing_in_progress (bool): Flag indicating if processing is ongoing.
        Returns:
            bool: True if buffers are full and ready for feature extraction, False otherwise.
        """
        try:
            # Update the buffers with raw data
            for channel in channel_names:
                if channel in raw_data:
                    primary_buffer[channel].append(raw_data[channel])

            # Log the current sizes of the primary buffer
            buffer_sizes = {channel: len(primary_buffer[channel]) for channel in channel_names}
            #self.logger.info(f"[update_eeg_buffers] Primary buffer sizes: {buffer_sizes}")

            # Check if the primary buffer is full
            if len(primary_buffer[channel_names[0]]) >= self.required_buffer_size:
                # Move data to secondary buffer for processing
                for channel in channel_names:
                    secondary_buffer[channel] = list(primary_buffer[channel])[:self.required_buffer_size]
                    primary_buffer[channel] = list(primary_buffer[channel])[self.required_buffer_size:]

                # Validate secondary buffer
                if all(len(secondary_buffer[channel]) == self.required_buffer_size for channel in channel_names):
                    eeg_data = np.array([list(secondary_buffer[channel]) for channel in channel_names])
                    #self.logger.info(f"[update_eeg_buffers] Secondary buffer filled. EEG data shape: {eeg_data.shape}")
                    self.process_and_extract_features(eeg_data)  # Extract features and update the feature window
                    return True
                else:
                    self.logger.warning("[update_eeg_buffers] Secondary buffer is not fully populated. Skipping processing.")
            return False
        except Exception as e:
            self.logger.error(f"Error updating EEG buffers: {e}")
            return False

    def preprocess_eeg_data(self):
        """
        Preprocess EEG data using optimized preprocessing techniques.
        """
        try:
            #self.logger.info("[Preprocess EEG Data] Starting preprocessing...")
            self.processing_in_progress = True

            # Retrieve data from the secondary buffer
            eeg_data = np.array([list(self.secondary_buffer[channel]) for channel in self.channel_names])
            if eeg_data.shape[1] == 0:
                #self.logger.error("[Preprocess EEG Data] Secondary buffer is empty. Skipping preprocessing.")
                self.processing_in_progress = False
                return None

            #self.logger.info(f"[Preprocess EEG Data] Raw EEG data shape: {eeg_data.shape}")

            # Check if the buffer is sufficiently filled
            if self.reference_channel:
                noise_ref = np.array(self.secondary_buffer[self.reference_channel])  # Use dedicated reference channel
            else:
                noise_ref = np.mean(eeg_data, axis=0, keepdims=True)  # Estimate noise from EEG data

            # 1. Noise Removal (Powerline + Band Filtering)
            eeg_data = apply_notch_filter(eeg_data, fs=self.fs)  # Remove powerline noise (50/60 Hz)
            #self.logger.info(f"[Preprocess EEG Data] After Notch Filter shape: {eeg_data.shape}")

            eeg_data = apply_bandpass_filter(eeg_data, lowcut=1.0, highcut=50.0, sampling_rate=self.fs)  # Retain relevant EEG bands
            #self.logger.info(f"[Preprocess EEG Data] After Bandpass Filter shape: {eeg_data.shape}")

            # 2. Noise Reduction & Re-referencing
            eeg_data = common_average_reference(eeg_data)  # Reduce background noise using CAR
            #self.logger.info(f"[Preprocess EEG Data] After Common Average Reference shape: {eeg_data.shape}")

            # 3. Adaptive Noise Cancellation (ANC) - If a noise reference exists
            if self.use_anc:
                eeg_data = apply_anc(eeg_data, noise_ref)  # Remove structured noise
                #self.logger.info(f"[Preprocess EEG Data] After ANC shape: {eeg_data.shape}")

            # 4. Independent Component Analysis (ICA) - Optional for real-time processing
            if self.use_ica:
                eeg_data = apply_ica(eeg_data)  # Remove eye blink & muscle artifacts
                #self.logger.info(f"[Preprocess EEG Data] After ICA shape: {eeg_data.shape}")

            # 5. Apply Smoothing - Hanning Window
            eeg_data = apply_hanning_window(eeg_data)  # Smooth signal for better feature extraction
            #self.logger.info(f"[Preprocess EEG Data] After Hanning Window shape: {eeg_data.shape}")

            # 6. Denoising using Discrete Wavelet Transform (DWT) - Optional for real-time use
            if self.use_dwt:
                eeg_data = apply_dwt_denoising(eeg_data)  # Reduce high-frequency noise
                #self.logger.info(f"[Preprocess EEG Data] After DWT Denoising shape: {eeg_data.shape}")

           

            self.processing_in_progress = False
            return eeg_data

        except Exception as e:
            self.logger.error(f"[Preprocess EEG Data] Error during preprocessing: {e}")
            self.processing_in_progress = False
            return None


    def extract_features(self, eeg_filtered):
        """
        Extract features using all available feature extraction methods.
        :param eeg_filtered: Preprocessed EEG data of shape (14, 256).
        :return: Feature vector of shape (14, total_features).
        """
        try:
            #self.logger.info("[Extract Features] Starting feature extraction...")
            band_power_features = []
            hjorth_features = []
            entropy_features = []
            fractal_features = []

            for i, channel in enumerate(self.channel_names):
                #self.logger.info(f"[Extract Features] Processing channel: {channel}")

                # Ensure the channel data is 1D
                channel_data = eeg_filtered[i]
                if len(channel_data.shape) != 1:
                    raise ValueError(f"Channel {channel} data is not 1D. Shape: {channel_data.shape}")

                # Compute band power
                band_power = compute_band_power(channel_data, fs=self.fs)
                band_power_array = np.array(list(band_power.values()))
                band_power_features.append(band_power_array)

                # Compute Hjorth parameters
                hjorth = compute_hjorth_parameters(channel_data)
                hjorth_features.append(hjorth)

                # Compute spectral entropy
                spectral_entropy = compute_spectral_entropy(channel_data, self.fs)
                entropy_features.append(spectral_entropy)

                # Compute Higuchi fractal dimension
                fractal_dimension = higuchi_fractal_dimension(channel_data)
                fractal_features.append(fractal_dimension)

            # Convert lists to numpy arrays
            band_power_features = np.array(band_power_features)  # (14, num_bands)
            #self.logger.info(f"[Extract Features] Band power features shape: {band_power_features.shape}")
            hjorth_features = np.array(hjorth_features)          # (14, 2)
            #self.logger.info(f"[Extract Features] Hjorth features shape: {hjorth_features.shape}")
            entropy_features = np.array(entropy_features)        # (14,)
            #self.logger.info(f"[Extract Features] Spectral entropy features shape: {entropy_features.shape}")
            fractal_features = np.array(fractal_features)        # (14,)
            #self.logger.info(f"[Extract Features] Fractal dimension features shape: {fractal_features.shape}")

            # Compute Temporal Derivatives
            first_order_derivatives = np.diff(eeg_filtered, axis=1, prepend=eeg_filtered[:, :1])  # (14, 256)
            #self.logger.info(f"[Extract Features] First order derivatives shape: {first_order_derivatives.shape}")
            second_order_derivatives = np.diff(first_order_derivatives, axis=1, prepend=first_order_derivatives[:, :1])  # (14, 256)
            #self.logger.info(f"[Extract Features] Second order derivatives shape: {second_order_derivatives.shape}")

            # Repeat static features along the time axis
            band_power_features = np.repeat(band_power_features, self.fs, axis=1)  # (14, num_bands * 256)
            #self.logger.info(f"[Extract Features] Repeated band power features shape: {band_power_features.shape}")
            hjorth_features = np.repeat(hjorth_features, self.fs, axis=1)          # (14, 3 * 256)
            #self.logger.info(f"[Extract Features] Repeated Hjorth features shape: {hjorth_features.shape}")
            entropy_features = np.repeat(np.expand_dims(entropy_features, axis=1), self.fs, axis=1)  # (14, 256)
            #self.logger.info(f"[Extract Features] Repeated spectral entropy features shape: {entropy_features.shape}")
            fractal_features = np.repeat(np.expand_dims(fractal_features, axis=1), self.fs, axis=1)  # (14, 256)
            #self.logger.info(f"[Extract Features] Repeated fractal dimension features shape: {fractal_features.shape}")

            # Concatenate all features
            features = np.concatenate((
                eeg_filtered,                     # (14, 256)
                first_order_derivatives,          # (14, 256)
                second_order_derivatives,         # (14, 256)
                band_power_features,              # (14, num_bands * 256)
                hjorth_features,                  # (14, 3 * 256)
                entropy_features,                 # (14, 256)
                fractal_features                  # (14, 256)
            ), axis=1)

            #self.logger.info(f"[Extract Features] Feature vector shape: {features.shape}")
            return features

        except Exception as e:
            self.logger.error(f"[Extract Features] Error during feature extraction: {e}")
            return None

    def extract_features_sequence(self, eeg_data_sequence):
        """
        Extract features for a sequence of EEG data (10 seconds).
        :param eeg_data_sequence: EEG data of shape (10, 14, 256).
        :return: Feature sequence of shape (10, 3584).
        """
        feature_sequence = []
        for eeg_data in eeg_data_sequence:
            features = self.extract_features(eeg_data)  # Extract features for 1 second
            feature_sequence.append(features.flatten())  # Flatten to (3584,)
        return np.array(feature_sequence)  # Shape: (10, 3584)

    def get_10_second_window(self):
        """
        Retrieve the last 10 seconds of EEG data for processing.
        :return: EEG data of shape (10, 14, 256).
        """
        if all(len(self.primary_buffer[channel]) >= 2560 for channel in self.channel_names):
            eeg_data_sequence = np.array([
                list(self.primary_buffer[channel])[-2560:].reshape(10, 256)
                for channel in self.channel_names
            ])
            return eeg_data_sequence.transpose(1, 0, 2)  # Shape: (10, 14, 256)
        else:
            self.logger.warning("Not enough data for a 10-second window.")
            return None

    def are_buffers_full(self, buffer_type="primary"):
            """
            Check if all buffers are full.
            Args:
                buffer_type (str): "primary" or "secondary".
            Returns:
                bool: True if all buffers are full, False otherwise.
            """
            buffer = self.primary_buffer if buffer_type == "primary" else self.secondary_buffer
            is_full = all(len(buffer[channel]) >= self.buffer_size for channel in self.channel_names)
            #self.logger.info(f"[Are Buffers Full] Buffer sizes: { {channel: len(buffer[channel]) for channel in self.channel_names} }")
            #self.logger.info(f"[Are Buffers Full] All buffers full: {is_full}")
            return is_full



    

    def train_rf_model(self):
        """Retrains the Random Forest model on the latest data."""
        if len(self.feature_history) < 50:
            return  # Wait until enough data is collected

        try:
            feature_matrix = np.array(self.feature_history)
            label_array = np.array(self.labels)
            self.rf_model.fit(feature_matrix, label_array)
            self.is_trained = True
            #self.logger.info("Live update: Random Forest retrained.")

        except Exception as e:
            self.logger.error(f"Error training Random Forest: {e}")

    def start_streaming(self):
        """
        Continuously read data from the EEG headset and populate buffers.
        """
        self.logger.info("Starting EEG data streaming...")
        loop_count = 0  # Initialize loop counter

        if not self.connect():
            #self.logger.error("Failed to connect to the EEG device. Exiting streaming process.")
            return

        try:
            while True:
                loop_count += 1
                #self.logger.info(f"[Streaming Loop] Iteration: {loop_count}")
                data = self.read_emotiv_data()
                if data:
                    # Log the Gyro data being added to the visualizer
                    if 'gyro_x' in data and 'gyro_y' in data:
                        #self.logger.info(f"Adding Gyro data to visualizer: gyro_x={data['gyro_x']}, gyro_y={data['gyro_y']}")
                        visualizer.update_gyro_data(data['gyro_x'], data['gyro_y'])

                    # Update EEG buffers
                    buffers_full = self.update_eeg_buffers(data)

                    # Check if buffers are full and process data
                    if buffers_full:
                        #self.logger.info("Buffers are full. Proceeding to data processing stage.")
                        processed_data = self.preprocess_eeg_data()
                        if processed_data is not None:
                            self.logger.info("Data processing complete. Proceeding to prediction.")
                           
                else:
                    self.logger.warning("No data received. Retrying...")
                    time.sleep(0.01)  # Small delay to avoid busy-waiting

        except KeyboardInterrupt:
            self.logger.info("Ctrl+C detected. Shutting down...")
        finally:
            self.disconnect()

    def get_latest_data(self):
        """
        Retrieves the latest EEG data from the buffers.
        Returns:
            dict: A dictionary containing the latest EEG data for each channel.
        """
        start_time = time.time()  # Record the start time
        timeout = 10  # Timeout in seconds to prevent infinite looping

        while not self.is_buffer_ready:
            #self.logger.info("Waiting for buffers to be fully populated...")
            self.is_buffer_ready = self.are_buffers_full("primary")  # Check buffer status
            if time.time() - start_time > timeout:
                #self.logger.error("Timeout while waiting for buffers to be fully populated.")
                return None  # Return None if timeout occurs
            time.sleep(0.1)  # Small delay to prevent busy-waiting

        # Retrieve the latest data
        latest_data = {channel: self.primary_buffer[channel][-1] if len(self.primary_buffer[channel]) > 0 else 0
                       for channel in self.channel_names}
        self.logger.info(f"Latest EEG data retrieved: {latest_data, len(latest_data)}")
        self.logger.info(f"Latest data shape: {np.array(list(latest_data.values())).shape}")
        return latest_data

    def get_buffer_data(self, buffer_type="primary"):
        """
        Retrieve data from the specified buffer.
        Args:
            buffer_type (str): "primary" or "secondary".
        Returns:
            dict: A dictionary containing buffer data for each channel.
        """
        buffer = self.primary_buffer if buffer_type == "primary" else self.secondary_buffer
        buffer_data = {channel: list(buffer[channel]) for channel in self.channel_names}
        self.logger.info(f"Retrieved {buffer_type} buffer data: {[(ch, len(data)) for ch, data in buffer_data.items()]}")
        return buffer_data

    def get_buffer_sizes(self):
        """
        Retrieve the current sizes of the primary and secondary buffers for each channel.
        Returns:
            dict: A dictionary containing the sizes of the buffers for each channel.
        """
        buffer_sizes = {
            "primary": {channel: len(self.primary_buffer[channel]) for channel in self.channel_names},
            "secondary": {channel: len(self.secondary_buffer[channel]) for channel in self.channel_names}
        }
        self.logger.info(f"Buffer sizes: {buffer_sizes}")
        return buffer_sizes

    @staticmethod
    def test_data_processing():
        """
        Standalone method to test the data processing pipeline with sample EEG data.
        This method will load data from an Excel file in the /data directory.
        """
        try:
            # Load data from Excel file
            file_path = "/Users/kushalpagolu/Documents/Code/epoch_tello_RL_3DBrain/data/Combined_EEG_Gyro_Data.xlsx"
            data = pd.read_excel(file_path)

            # Ensure the file contains the required columns
            required_columns = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"Excel file must contain the following columns: {required_columns}")

            # Extract a single row of data for testing
            sample_data = data.iloc[0][required_columns].to_numpy()
            print(f"Sample Data Shape: {sample_data.shape}")

            # Apply preprocessing techniques
            print("\n--- Applying Preprocessing Techniques ---")
            eeg_data = apply_notch_filter(sample_data, fs=256)
            print(f"After Notch Filter Shape: {eeg_data.shape}")

            eeg_data = apply_bandpass_filter(eeg_data, lowcut=1.0, highcut=50.0, sampling_rate=256)
            print(f"After Bandpass Filter Shape: {eeg_data.shape}")

            eeg_data = common_average_reference(eeg_data)
            print(f"After Common Average Reference Shape: {eeg_data.shape}")

            eeg_data = apply_ica(eeg_data)
            print(f"After ICA Shape: {eeg_data.shape}")

            eeg_data = apply_hanning_window(eeg_data)
            print(f"After Hanning Window Shape: {eeg_data.shape}")

            eeg_data = apply_dwt_denoising(eeg_data)
            print(f"After DWT Denoising Shape: {eeg_data.shape}")

            # Extract features
            print("\n--- Extracting Features ---")
            band_power_features = compute_band_power(eeg_data, fs=256)
            print(f"Band Power Features: {band_power_features}")

            hjorth_features = compute_hjorth_parameters(eeg_data)
            print(f"Hjorth Parameters: {hjorth_features}")

            spectral_entropy = compute_spectral_entropy(eeg_data, fs=256)
            print(f"Spectral Entropy: {spectral_entropy}")

            fractal_dimension = higuchi_fractal_dimension(eeg_data)
            print(f"Higuchi Fractal Dimension: {fractal_dimension}")

            # Combine features into a feature vector
            feature_vector = np.concatenate((
                list(band_power_features.values()),
                hjorth_features,
                [spectral_entropy],
                [fractal_dimension]
            ))
            print(f"Feature Vector: {feature_vector}, Shape: {feature_vector.shape}")

        except Exception as e:
            print(f"Error in test_data_processing: {e}")

    def add_data(self, data):
        """
        Add raw EEG data to the buffers for visualization.
        """
        if isinstance(data, dict):  # Assuming data is a dictionary with channel names as keys
            for i, channel in enumerate(self.channel_names):
                if channel in data:
                    self.data_buffers[i].append(data[channel])
        elif isinstance(data, list):  # Handle raw data as a list
            for i in range(min(len(data), self.num_channels)):
                self.data_buffers[i].append(data[i])
        else:
            self.logger.warning("Unsupported data format for visualization.")  # Use self.logger instead of logging.warning

    def update_feature_window(self, feature_vector):
        """
        Add a 1-second feature vector to the 10-second feature window.
        """
        self.feature_window.append(feature_vector)
        #self.logger.info(f"[update_feature_window] Updated feature window. Current size: {len(self.feature_window)}")

    def get_feature_sequence(self):
        """
        Retrieve the 10-second feature sequence if the window is full.
        :return: Feature sequence of shape (10, feature_vector_length) or None.
        """
        if len(self.feature_window) == 10:
            #self.logger.info("Feature window is full. Returning feature sequence.")
            return np.array(self.feature_window)  # Shape: (10, feature_vector_length)
        else:
            #self.logger.info(f"Feature window is not full. Current size: {len(self.feature_window)}")
            return None

    def process_and_extract_features(self, eeg_data):
        """
        Process raw EEG data, extract features, and update the feature window.
        """
        try:
            # Preprocess the EEG data
            preprocessed_data = self.preprocess_eeg_data()
            if preprocessed_data is not None:
                # Extract features for 1 second
                feature_vector = self.extract_features(preprocessed_data).flatten()  # Shape: (feature_vector_length,)
                if feature_vector is not None:
                    self.update_feature_window(feature_vector)  # Update the feature window
                    self.logger.info(f"[process_and_extract_features] Feature vector added. Feature window size: {len(self.feature_window)}")
                else:
                    self.logger.warning("[process_and_extract_features] Feature vector is None. Skipping update to feature window.")
                return feature_vector
            else:
                self.logger.warning("[process_and_extract_features] Preprocessed data is None. Skipping feature extraction.")
                return None
        except Exception as e:
            self.logger.error(f"Error during feature extraction: {e}")
            return None

    def get_observation(self):
        """
        Retrieves the latest processed observation from the feature window.
        Returns:
            np.ndarray: The latest observation of shape (feature_vector_length,) or None if the feature window is empty.
        """
        try:
            if len(self.feature_window) > 0:
                latest_observation = np.array(self.feature_window[-1])  # Get the most recent feature vector
                self.logger.info(f"[get_observation] Retrieved latest observation. Shape: {latest_observation.shape}")
                return latest_observation
            else:
                self.logger.warning("[get_observation] Feature window is empty. Returning None.")
                return None
        except Exception as e:
            self.logger.error(f"[get_observation] Error retrieving observation: {e}")
            return None


if __name__ == "__main__":
    streamer = EmotivStreamer()
    streamer.test_data_processing()
    if streamer.connect():
        try:
            streamer.start_streaming()
        except KeyboardInterrupt:
            streamer.logger.info("Streaming stopped by user.")
        finally:
            streamer.disconnect()
    else:
        streamer.logger.error("Failed to connect to the EEG device.")
