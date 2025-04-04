import logging
import os
import time
import signal
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import threading
from kalman_filter import KalmanFilter  # Assuming you have a kalman_filter module

# Global variables for human feedback
human_feedback = None
feedback_condition = threading.Condition()

def signal_handler(sig, frame, env, emotiv, data_store, stop_saving_thread, stop_main_loop, MODEL_FILENAME):
    logger = logging.getLogger(__name__)
    logger.info("Ctrl+C detected. Shutting down...")
    stop_saving_thread.set()  # Stop the save_data_continuously thread
    #logger.info("Signaled save_data_continuously thread to stop.")
    stop_main_loop.set()  # Stop other threads
    #logger.info("Signaled main loop to stop.")

    # Log the size of data_store
    #logger.info(f"Data store size at shutdown: {len(data_store)}")

    # Ensure the model is saved before exit
    if hasattr(env, 'model') and env.model:
        logger.info(f"Saving model before exiting...")
        env.model.save(MODEL_FILENAME)
        logger.info(f"Model saved to {MODEL_FILENAME}.zip")

    if emotiv.device:
        emotiv.disconnect()

    # Save remaining data before exiting
    if data_store:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        raw_filename = os.path.join("data", f"EEG_Raw_{timestamp}.xlsx")
        processed_filename = os.path.join("data", f"Processed_Data_{timestamp}.xlsx")

        try:
            # Define EEG and gyro channel names
            eeg_channels = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
            gyro_channels = ["gyro_x", "gyro_y"]
            column_names = eeg_channels + gyro_channels

            # Convert data_store to a DataFrame
            df = pd.DataFrame(data_store, columns=column_names)

            # Compute volts for EEG channels
            for channel in eeg_channels:
                df[f"{channel}_volts"] = df[channel] * 0.51

            # Apply Kalman filter to gyro data
            kalman_filter_x = KalmanFilter()
            kalman_filter_y = KalmanFilter()
            df["gyro_x_deg_s"] = df["gyro_x"].apply(kalman_filter_x.update)
            df["gyro_y_deg_s"] = df["gyro_y"].apply(kalman_filter_y.update)

            # Integrate gyro data for angles
            df["head_roll_deg"] = df["gyro_x_deg_s"].cumsum() * (1 / 128)
            df["head_pitch_deg"] = df["gyro_y_deg_s"].cumsum() * (1 / 128)

            # Median subtraction for EEG channels
            for channel in eeg_channels:
                df[f"{channel}_med_subtracted"] = df[channel].subtract(df[eeg_channels].median(axis=1), axis=0)

            # Clip and smooth EEG data
            for i in range(1, len(df)):
                delta = df[eeg_channels].iloc[i] - df[eeg_channels].iloc[i-1]
                delta = delta.clip(-15, 15)
                df.loc[i, eeg_channels] = df.loc[i-1, eeg_channels] + delta

            # Save raw data
            df[eeg_channels + gyro_channels].to_excel(raw_filename, index=False)
            #logger.info(f"Raw Data saved to {raw_filename}")

            # Save processed data
            df.to_excel(processed_filename, index=False)
            #logger.info(f"Processed Data saved to {processed_filename}")
        except Exception as e:
            logger.error(f"Error saving data to Excel: {str(e)}")

    plt.close('all')
    exit(0)

def feedback_signal_handler(sig, frame):
    """
    Signal handler for human feedback.
    SIGUSR1: Approve action.
    SIGUSR2: Reject action.
    """
    global human_feedback
    with feedback_condition:
        if sig == signal.SIGUSR1:
            human_feedback = True
            logging.info("SIGUSR1 received: Approve action.")
        elif sig == signal.SIGUSR2:
            human_feedback = False
            logging.info("SIGUSR2 received: Reject action.")
        else:
            logging.warning(f"Unexpected signal received: {sig}")
        feedback_condition.notify()  # Notify waiting threads
        logging.info("feedback_condition.notify() called.")
