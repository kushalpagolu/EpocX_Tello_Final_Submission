import os
import time
import pandas as pd
from datetime import datetime
import logging
import threading
from kalman_filter import KalmanFilter

logger = logging.getLogger(__name__)

def save_data_continuously(data_store, stop_saving_thread):
    """
    Continuously saves data from the data_store to an Excel file every 30 seconds.
    Adds computed EEG volt columns to the saved data.
    """
    logging.info("[Data Saver] save_data_continuously thread started.")
    try:
        eeg_channels = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]

        # Ensure the "data" directory exists
        os.makedirs("data", exist_ok=True)

        while not stop_saving_thread.is_set():
            if data_store:
                #logging.info(f"[Data Saver] Saving {len(data_store)} packets.")
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                raw_filename = os.path.join("data", f"EEG_Raw_{timestamp}.xlsx")
                processed_filename = os.path.join("data", f"Processed_Data_{timestamp}.xlsx")

                try:
                    # Convert data_store to a DataFrame with proper column names
                    df = pd.DataFrame(data_store, columns=eeg_channels + ["gyro_x", "gyro_y"])

                    # Compute volts for EEG channels
                    for channel in eeg_channels:
                        df[f"{channel}_volts"] = df[channel] * 0.51


                    # Apply Kalman filter to gyro data
                    kalman_filter_x = KalmanFilter()
                    kalman_filter_y = KalmanFilter()

                    # Compute gyro data and integrate for angles
                    df["gyro_x_deg_s"] = df["gyro_x"].apply(kalman_filter_x.update)
                    df["gyro_y_deg_s"] = df["gyro_y"].apply(kalman_filter_x.update)
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

                    # Save raw and processed data
                    df[eeg_channels + ["gyro_x", "gyro_y"]].to_excel(raw_filename, index=False)
                    #logger.info(f"Raw Data saved to {raw_filename}")
                    df.to_excel(processed_filename, index=False)
                    #logger.info(f"Processed Data saved to {processed_filename}")

                    # Clear the data_store after saving
                    data_store.clear()
                except Exception as e:
                    logger.error(f"Error saving data to Excel: {str(e)}")
            else:
                logger.info("Data store is empty. Skipping save operation.")

            time.sleep(30)
    except Exception as e:
        logging.error(f"[Data Saver] Error in save_data_continuously: {e}")
    finally:
        logging.info("[Data Saver] save_data_continuously thread stopping.")
