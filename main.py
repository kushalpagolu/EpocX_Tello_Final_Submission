import threading
import queue  # For decoupling streaming and preprocessing
from datetime import datetime
import logging
import argparse
import signal  # Add this import
from signal_handler import signal_handler  # Import refactored signal handler
from stream_data import EmotivStreamer
from learning_rlagent import DroneControlEnv, input_listener  # Import refactored input listener
from model_utils import load_or_create_model
from lstm_handler import LSTMHandler
from visualizer_realtime3D import RealtimeEEGVisualizer
import time
import os
import pandas as pd
from data_saver import save_data_continuously  # Import refactored data saver
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import numpy
import logging

logging.getLogger('matplotlib').setLevel(logging.WARNING)

MODEL_FILENAME = "/Users/kushalpagolu/Desktop/EmotivEpoch/epoch_tello_RL_3DBrain/models/drone_rl_eeg_human_learning.zip"
stop_saving_thread = threading.Event()
stop_main_loop = threading.Event()
data_queue = queue.Queue()  # Shared queue for streaming and preprocessing
data_store = []  # Initialize data store
visualization_queue = queue.Queue()  # Queue for sending data to the visualizer
visualization_ready = threading.Event()  # Event to signal when visualization can start

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')



def preprocessing_thread(data_queue, env, model_agent, visualizer, emotiv):
    """
    Thread for preprocessing data and making predictions in chunks of 256 frames.
    """
    lstm_handler = LSTMHandler()
    lock = threading.Lock()

    while not stop_main_loop.is_set():
        try:
            # Collect a single packet from the queue
            packet = data_queue.get(timeout=1)
            if packet and len(packet) > 0:
                #logging.info(f"[Preprocessing Thread] Packet received. Packet keys: {list(packet.keys())}")

                # Update EEG buffers
                buffers_full = emotiv.update_eeg_buffers(packet, emotiv.channel_names, emotiv.primary_buffer, emotiv.secondary_buffer, emotiv.processing_in_progress)

                # Check if the feature window is ready
                feature_sequence = emotiv.get_feature_sequence()
                if feature_sequence is not None:
                    logging.info(f"[Preprocessing Thread] 10-second feature sequence shape: {feature_sequence.shape}")

                    # Predict using the LSTM model
                    with lock:
                        lstm_output = lstm_handler.predict_sequence(feature_sequence)
                        if lstm_output is not None:
                            #logging.info(f"[Preprocessing Thread] LSTM prediction complete. Output shape: {lstm_output.shape}")
                            try:
                                # Predict action using the RL agent
                                action, _ = model_agent.predict(lstm_output, deterministic=False)
                                logging.info(f"[Preprocessing Thread] Predicted action RL Agent: {action}")

                                # Pass the action to the environment's step function
                                state, reward, done, info = env.step(action)
                                logging.info(f"[Preprocessing Thread] Step executed. Reward: {reward}, Done: {done}, Info: {info}")

                            except Exception as e:
                                logging.error(f"[Preprocessing Thread] Error during RL agent prediction or step execution: {e}")
                        else:
                            logging.warning("[Preprocessing Thread] LSTM prediction returned None. Skipping...")
                else:
                    logging.info("[Preprocessing Thread] Feature window is not full. Waiting for more data.")
            else:
                logging.warning("[Preprocessing Thread] Invalid or empty packet received. Skipping...")

        except queue.Empty:
            logging.warning("[Preprocessing Thread] Data queue is empty. Waiting for new packets...")
        except Exception as e:
            logging.error(f"[Preprocessing Thread] Error during preprocessing: {e}")

def streaming_thread(emotiv, data_queue, visualization_queue):
    """
    Thread for streaming data from the Emotiv device.
    """
    valid_packet_count = 0
    empty_packet_count = 0
    loop_count = 0  # Initialize loop counter

    while not stop_main_loop.is_set():
        loop_count += 1
        start_time = time.time()  # Start timing
        #logging.info(f"[Streaming Thread] Iteration: {loop_count}")
        try:
            packet = emotiv.read_emotiv_data()
            #logging.debug(f"[Streaming Thread] Raw packet data: {packet}")
            if packet and len(packet) > 0:  # Ensure packet is valid and non-empty
                valid_packet_count += 1
                #logging.info(f"[Streaming Thread] Valid packet received. Count: {valid_packet_count}, Packet: {packet}")
                data_queue.put(packet)  # Add packet to the preprocessing queue
                #logging.info(f"[Streaming Thread] Packet added to data_queue. Queue size: {data_queue.qsize()}")
                visualization_queue.put(packet)  # Send raw data to the visualization queue
                visualization_ready.set()  # Signal that visualization can start
                #logging.info(f"[Streaming Thread] Packet added to visualization_queue.")
                if 'gyro_x' in packet and 'gyro_y' in packet:
                    # Call update_gyro_data to update the gyro buffers
                    visualizer.update_gyro_data(packet['gyro_x'], packet['gyro_y'])
                    #logging.info(f"[Streaming Thread] Gyro data updated: gyro_x={packet['gyro_x']}, gyro_y={packet['gyro_y']}")
            else:
                empty_packet_count += 1
                #logging.warning(f"[Streaming Thread] Empty or invalid packet skipped. Count: {empty_packet_count}")
                if empty_packet_count > 200:
                    logging.error("[Streaming Thread] Too many empty packets. Stopping streaming and reconnecting... empty_packet_count")
                    emotiv.disconnect()
                    time.sleep(3)  # Wait before reconnecting
                    if not emotiv.connect():
                        logging.error("[Streaming Thread] Failed to reconnect to Emotiv device.")
                        emotiv.connect()
                    empty_packet_count = 0  # Reset empty packet count after reconnecting
                continue


        except Exception as e:
            logging.error(f"[Streaming Thread] Error while reading data: {e}")
        end_time = time.time()  # End timing
        #logging.info(f"[Streaming Thread] Time taken for iteration: {end_time - start_time:.4f} seconds")
        time.sleep(0.01)  # Small delay to avoid busy-waiting


def run_visualization_on_main_thread(visualizer, visualization_queue):
    """
    Run the visualization logic on the main thread.
    """
    def update_visualization():
        while not visualization_queue.empty():
            packet = visualization_queue.get()
            visualizer.add_data(packet)  # Add data to the visualizer
        visualizer.update_2d(None)  # Update the 2D visualization

    # Wait for the streaming thread to signal that data is ready
    logging.info("[Visualization] Waiting for the first data packet...")
    visualization_ready.wait()  # Block until the event is set
    logging.info("[Visualization] First data packet received. Starting visualization.")

    # Start the visualization
    anim_2d = FuncAnimation(
        visualizer.fig_2d,
        lambda frame: update_visualization(),
        interval=100,
        cache_frame_data=False  # Disable frame data caching to suppress the warning
    )
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG Data Streamer and Drone Control")
    parser.add_argument("--connect-drone", action="store_true", help="Connect to the drone")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    emotiv = EmotivStreamer()
    visualizer = RealtimeEEGVisualizer()
    env = DroneControlEnv(connect_drone=args.connect_drone)

    # Start the input listener thread
    input_thread = threading.Thread(target=input_listener, daemon=True)
    input_thread.start()

    # Initialize LSTMHandler separately if not part of DroneControlEnv
    model_lstm = LSTMHandler()  # Initialize the LSTMHandler
    model_agent = load_or_create_model(env)

    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(
        sig, frame, env, emotiv, data_store, stop_saving_thread, stop_main_loop, MODEL_FILENAME
    ))

    if emotiv.connect():
        logging.info("Emotiv EEG device connected. Starting real-time EEG streaming.")
        try:
            # Start background thread for data saving
            logging.info("[Main] Starting save_data_continuously thread.")
            save_thread = threading.Thread(target=save_data_continuously, args=(data_store, stop_saving_thread))
            save_thread.daemon = True
            save_thread.start()

            # Start streaming and preprocessing threads
            stream_thread = threading.Thread(target=streaming_thread, args=(emotiv, data_queue, visualization_queue))
            preprocess_thread = threading.Thread(target=preprocessing_thread, args=(data_queue, env, model_agent, visualizer, emotiv))
            stream_thread.start()
            preprocess_thread.start()

            # Run visualization on the main thread
            run_visualization_on_main_thread(visualizer, visualization_queue)
        except Exception as e:
            logging.error(f"An error occurred: {e}")
        finally:
            logging.info("[Main] Stopping all threads.")
            stop_saving_thread.set()  # Signal the save_data_continuously thread to stop
            stop_main_loop.set()  # Signal other threads to stop
            stream_thread.join()
            preprocess_thread.join()
            save_thread.join()  # Ensure save_data_continuously thread stops
            logging.info("[Main] All threads stopped.")

            # Log model parameters to a file
            logging.info("Logging LSTM model and RL agent parameters to a file...")
            env.log_model_parameters(model_lstm, model_agent)

            emotiv.disconnect()
    else:
        logging.error("Failed to connect to Emotiv device.")
