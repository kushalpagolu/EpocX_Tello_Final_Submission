import gym
import numpy as np
from gym import spaces
import os
import logging
import time
import threading
from datetime import datetime
from stream_data import EmotivStreamer
from model_utils import load_or_create_model  # Updated import for model utilities
from drone_control import TelloController  # Updated import for TelloController
from lstm_handler import LSTMHandler  # Import the LSTMHandler class
import signal
from signal_handler import feedback_signal_handler, feedback_condition
import select
import sys
import queue
import random
import torch

# Generate a timestamped log file name
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f"/Users/kushalpagolu/Desktop/EmotivEpoch/epoch_tello_RL_3DBrain/logs/rl_agent_{timestamp}.log"


# Queue for handling user input
input_queue = queue.Queue()


# Update logging configuration to save logs to a new file for each run
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),  # Save logs to a timestamped file
        logging.StreamHandler()  # Also display logs in the console
    ]
)

logger = logging.getLogger(__name__)

MODEL_FILENAME = os.path.join(os.getcwd(), "models", "drone_rl_eeg_human_loop")
ACTION_DELAY = 5  # Seconds between actions
HUMAN_FEEDBACK_TIMEOUT = 12  # Increased timeout for human feedback
MAX_SPEED = 30  # Maximum speed percentage
EXPECTED_OBSERVATION_SIZE = 5  # Define the expected observation size


def input_listener():
    """Captures user input and places it in a queue."""
    thread_logger = logging.getLogger(__name__)  # Ensure the thread uses the same logger
    thread_logger.info("Input listener thread started.")  # Log thread start
    while True:
        try:
            thread_logger.debug("Waiting for user input...")
            user_input = input("Enter 'y' to approve or 'n' to reject: ").strip().lower()  # Use input() for simplicity
            thread_logger.info(f"User input received: {user_input}")  # Log the input
            input_queue.put(user_input)
        except KeyboardInterrupt:
            thread_logger.info("Input listener interrupted by Ctrl+C. Exiting thread.")
            break  # Exit the loop gracefully
        except EOFError:
            thread_logger.warning("EOFError encountered in input_listener. Continuing...")
            continue  # Handle EOFError gracefully
        except Exception as e:
            thread_logger.error(f"Unexpected error in input_listener: {e}")


class DroneControlEnv(gym.Env):
    def __init__(self, connect_drone=False, max_speed=MAX_SPEED):
        self.logger = logging.getLogger(__name__)  # Initialize logger first
        super(DroneControlEnv, self).__init__()  # Re-add this line

        # Initialize the EEG processor
        self.eeg_processor = EmotivStreamer()
        # Define the observation space with the expected shape
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(EXPECTED_OBSERVATION_SIZE,), dtype=np.float32)
        # Define the action space as a single Box
        self.action_space = spaces.Box(
            low=np.array([0, -1.0, -1.0, -1.0, -1.0]),  # Discrete action (0-4) and continuous actions (-1.0 to 1.0)
            high=np.array([4, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )

        self.current_state = np.zeros(EXPECTED_OBSERVATION_SIZE)
        self.connect_drone = connect_drone  # Use the value passed during instantiation
        self.drone_controller = None  # Initialize to None
        self.drone_connected = False  # Explicitly initialize drone_connected to False
        if self.connect_drone:  # Check the actual value of connect_drone
            self.drone_controller = TelloController()
            self.connect_drone_controller()  # Ensure the drone is connected during initialization
        self.model = None
        self.last_action_time = 0
        self.max_speed = max_speed  # Set max speed
        self.has_taken_off = False  # Ensure takeoff is tracked
        self.model_lstm = LSTMHandler()  # Initialize the LSTMHandler


    def connect_drone_controller(self):
        if self.drone_controller and not self.drone_connected:
            self.drone_connected = self.drone_controller.connect()
            if self.drone_connected:
                self.logger.info("Drone connected successfully.")
            else:
                self.logger.error("Failed to connect to the drone.")
            return self.drone_connected

    def takeoff_drone(self):
        """Ensure the drone takes off before executing commands."""
        if not self.drone_controller:
            self.logger.error("Drone controller is not initialized. Cannot take off.")
            return

        if not self.drone_connected:
            self.logger.error("Drone is not connected. Cannot take off.")
            return

        if not self.has_taken_off:
            try:
                self.logger.info("Taking off the drone.")
                self.drone_controller.takeoff()
                self.has_taken_off = True
            except Exception as e:
                self.logger.error(f"Error during takeoff: {e}")

    def land_drone(self):
        """Ensure the drone lands safely."""
        if self.drone_controller and self.drone_connected and self.has_taken_off:
            try:
                self.logger.info("Landing the drone.")
                self.drone_controller.land()
                self.has_taken_off = False
            except Exception as e:
                self.logger.error(f"Error during landing: {e}")

    def step(self, action):
        """
        Executes a step in the environment.
        Args:
        action (np.ndarray): Flattened action array.
                - action[0]: Discrete action (0-8)
                - action[1:]: Continuous actions (4 values for velocities)
        Returns:
            tuple: (state, reward, done, info)
        """
        reward = 0
        done = False
        info = {}
        new_state = None  # Initialize new_state to avoid referencing before assignment

        # Ensure the drone is connected and has taken off
        if self.connect_drone:
            if not self.drone_connected:
                self.logger.error("Drone is not connected. Aborting step.")
                return self.current_state, 0, True, {"error": "Drone not connected"}
            self.takeoff_drone()

        # --- First Action: Takeoff ---
        if not self.has_taken_off:
            self.logger.info("First action detected. Overriding to 'Takeoff'.")
            action = np.array([5, 0, 0, 0, 0])  # Discrete action 5 corresponds to "Ascend"
            reward += 10  # Positive reward for successful takeoff
            self.takeoff_drone()  # Use the updated takeoff_drone method
            self.logger.info("Takeoff action executed. Reward updated.")

        # Validate the action
        if not isinstance(action, np.ndarray) or action.shape[0] != 5:
            self.logger.error(f"Invalid action format: {action}")
            return self.current_state, reward, done, info

        # Map the action to a command
        command = self._map_action_to_command(action)
        self.logger.info(f"Mapped Command: {command['command']}, Velocities: {command['velocities']}")

        # --- Human-in-the-Loop Confirmation ---
        self.logger.info(f"Approve action: {command['command']}? ('y' to approve, 'n' to reject, timeout={HUMAN_FEEDBACK_TIMEOUT}s)")
        human_feedback = self.get_human_feedback()

        if human_feedback is not None:
            if human_feedback:
                reward += 7  # Positive reward for human approval
                self.logger.info("Action approved by human.")
            else:
                reward -= 5  # Negative penalty for human rejection
                self.logger.info("Action rejected by human. Skipping action.")
                return self.current_state, reward, done, info
        else:
            reward -= 1  # Smaller penalty for no response within timeout
            self.logger.info("No feedback received. Assuming user is still thinking.")

        # Execute the movement command
        current_time = time.time()
        if current_time - self.last_action_time >= ACTION_DELAY:
            self.last_action_time = current_time
            if self.connect_drone and self.drone_connected:
                try:
                    self.drone_controller.send_rc_control(
                        command["velocities"]["left_right_velocity"],
                        command["velocities"]["forward_backward_velocity"],
                        command["velocities"]["up_down_velocity"],
                        command["velocities"]["yaw_velocity"]
                    )
                    self.logger.info(f"Drone action executed: {command['command']} with velocities "
                                    f"LR={command['velocities']['left_right_velocity']}, "
                                    f"FB={command['velocities']['forward_backward_velocity']}, "
                                    f"UD={command['velocities']['up_down_velocity']}, "
                                    f"YAW={command['velocities']['yaw_velocity']}")
                except Exception as e:
                    self.logger.error(f"Error sending RC control: {e}")
                    return self.current_state, -1, True, {"error": "Drone command failed"}
            else:
                self.logger.info(f"Simulating action: {command['command']} with velocities "
                                f"LR={command['velocities']['left_right_velocity']}, "
                                f"FB={command['velocities']['forward_backward_velocity']}, "
                                f"UD={command['velocities']['up_down_velocity']}, "
                                f"YAW={command['velocities']['yaw_velocity']}")
        else:
            self.logger.info(f"Action delayed. Remaining time: {ACTION_DELAY - (current_time - self.last_action_time):.2f} seconds")
            reward -= 0.02  # Small penalty for waiting

        # Simulate or fetch the new state
        new_state = self.eeg_processor.get_observation()  # Example: Fetch observation from EEG processor

        # Handle the case where new_state is None
        if new_state is None:
            self.logger.warning("New state is None. Buffers may not be fully populated. Returning current state.")
            return self.current_state, reward, done, info

        # Validate and correct the new state shape
        if new_state.shape != (EXPECTED_OBSERVATION_SIZE,):
            self.logger.warning(f"Observation shape mismatch: expected {EXPECTED_OBSERVATION_SIZE}, got {new_state.shape}. Padding or truncating.")
            if new_state.shape[0] < EXPECTED_OBSERVATION_SIZE:
                new_state = np.pad(new_state, (0, EXPECTED_OBSERVATION_SIZE - new_state.shape[0]), mode='constant')
            else:
                new_state = new_state[:EXPECTED_OBSERVATION_SIZE]

        self.update_state(new_state)

        return self.current_state, reward, done, info

    def get_human_feedback(self):
        """Waits for user input within the timeout period."""
        self.logger.info("Waiting for human feedback...")
        feedback = None
        start_time = time.time()
        while time.time() - start_time < HUMAN_FEEDBACK_TIMEOUT:
            try:
                self.logger.debug("Checking input queue for user input...")
                user_input = input_queue.get(timeout=3)  # Increase timeout to 1 second for better blocking
                self.logger.info(f"User input from queue: {user_input}")  # Log the input
                if user_input == 'y':
                    self.logger.info("Takeoff approved by user.")
                    return True
                elif user_input == 'n':
                    self.logger.info("Takeoff rejected by user.")
                    return False
                else:
                    self.logger.warning("Invalid input. Please press 'y' or 'n'.")
            except queue.Empty:
                self.logger.debug("No input received yet. Waiting...")
                time.sleep(1)  # Add a delay of 1 second before checking again
                continue  # No input yet, keep waiting
            except Exception as e:
                self.logger.error(f"Unexpected error in get_human_feedback: {e}")

        self.logger.info("No feedback received within timeout.")
        return None  # Return None if no feedback is received

    def close(self):
        """Ensure the drone lands safely when the environment is closed."""
        self.logger.info("Closing the environment.")
        self.land_drone()
        if self.drone_controller:
            self.logger.info("Disconnecting the drone.")
            self.drone_controller.tello.end()

    def _map_action_to_command(self, action):
        """
        Maps the action vector to Tello drone commands using its properties.
        Args:
            action (np.ndarray): Predicted action vector from the RL agent.
        Returns:
            dict: A dictionary containing the mapped Tello drone commands.
        """
        # Validate the action vector
        if not isinstance(action, np.ndarray) or action.shape[0] != 5:
            self.logger.error(f"Invalid action format: {action}")
            return {"command": "Invalid", "velocities": None}

        # Parse the discrete action and continuous values
        discrete_action = int(np.clip(round(action[0] * 8), 0, 8))  # Scale and clip to [0, 8]
        continuous_actions = action[1:]  # Continuous values for velocities

        # Scale continuous actions to the Tello drone's velocity range (-100 to 100)
        scaled_velocities = (np.tanh(continuous_actions) * 100).astype(int)

        left_right_velocity, forward_backward_velocity, up_down_velocity, yaw_velocity = scaled_velocities

        # Initialize the command dictionary
        command = {"command": "Hover", "velocities": {
            "left_right_velocity": 0,
            "forward_backward_velocity": 0,
            "up_down_velocity": 0,
            "yaw_velocity": 0
        }}

        # Map discrete actions to high-level commands
        if discrete_action == 0:  # Hover
            command["command"] = "Hover"
        elif discrete_action == 1:  # Move Forward
            command["command"] = "Move Forward"
            command["velocities"]["forward_backward_velocity"] = max(20, forward_backward_velocity)
        elif discrete_action == 2:  # Move Backward
            command["command"] = "Move Backward"
            command["velocities"]["forward_backward_velocity"] = min(-20, forward_backward_velocity)
        elif discrete_action == 3:  # Move Left
            command["command"] = "Move Left"
            command["velocities"]["left_right_velocity"] = min(-20, left_right_velocity)
        elif discrete_action == 4:  # Move Right
            command["command"] = "Move Right"
            command["velocities"]["left_right_velocity"] = max(20, left_right_velocity)
        elif discrete_action == 5:  # Ascend
            command["command"] = "Ascend"
            command["velocities"]["up_down_velocity"] = max(20, up_down_velocity)
        elif discrete_action == 6:  # Descend
            command["command"] = "Descend"
            command["velocities"]["up_down_velocity"] = min(-20, up_down_velocity)
        elif discrete_action == 7:  # Rotate Left
            command["command"] = "Rotate Left"
            command["velocities"]["yaw_velocity"] = min(-20, yaw_velocity)
        elif discrete_action == 8:  # Rotate Right
            command["command"] = "Rotate Right"
            command["velocities"]["yaw_velocity"] = max(20, yaw_velocity)

        # Log the mapped command for debugging
        self.logger.info(f"Discrete Action: {discrete_action}, Continuous Actions: {continuous_actions}")
        self.logger.info(f"Scaled Velocities: {scaled_velocities}")
        self.logger.info(f"Mapped Command: {command['command']}, Velocities: {command['velocities']}")

        return command

    def update_state(self, new_state):
        self.current_state = new_state

    def log_model_parameters(self, lstm_model, rl_model):
        """
        Logs the parameters of the LSTM model and the RL agent to a file.
        Args:
            lstm_model: The LSTM model instance.
            rl_model: The RL agent instance.
        """
        try:
            # Generate a timestamped log file name
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = f"/Users/kushalpagolu/Desktop/EmotivEpoch/epoch_tello_RL_3DBrain/logs/model_parameters_{timestamp}.log"

            with open(log_file, 'w') as f:
                # Log LSTM model parameters
                f.write("LSTM Model Parameters:\n")
                for name, param in lstm_model.named_parameters():
                    f.write(f"{name}: {param.data}\n")

                # Log RL agent model parameters
                f.write("\nRL Agent Model Parameters:\n")
                for name, param in rl_model.policy.named_parameters():
                    f.write(f"{name}: {param.data}\n")

            self.logger.info(f"Model parameters logged to {log_file}.")
        except Exception as e:
            self.logger.error(f"Error logging model parameters: {e}")

    def get_observation(self):
        """
        Processes raw EEG data to generate an observation for the RL agent.
        Returns:
            np.ndarray: Processed observation of shape (EXPECTED_OBSERVATION_SIZE,).
        """
        try:
            # Fetch raw EEG data from the EmotivStreamer
            raw_data = self.eeg_processor.read_emotiv_data()
            if (raw_data is None or len(raw_data) == 0):
                self.logger.warning("[get_observation] No raw data available from EEG streamer.")
                return None

            # Update EEG buffers and check if buffers are full
            buffers_full = self.eeg_processor.update_eeg_buffers(
                raw_data,
                self.eeg_processor.channel_names,
                self.eeg_processor.primary_buffer,
                self.eeg_processor.secondary_buffer,
                self.eeg_processor.processing_in_progress
            )
            if not buffers_full:
                self.logger.info("[get_observation] EEG buffers are not yet full. Waiting for more data.")
                return None

            # Extract feature sequence from the EEG buffers
            feature_sequence = self.eeg_processor.get_feature_sequence()
            if feature_sequence is None:
                self.logger.info(f"[get_observation] Feature window is not full. Current size: {len(self.eeg_processor.feature_window)}")
                return None

            # Predict using the LSTM model
            lstm_output = self.model_lstm.predict_sequence(feature_sequence)
            if lstm_output is None:
                self.logger.error("[get_observation] LSTM prediction failed. Returning None.")
                return None

            # Ensure the output matches the expected observation size
            if lstm_output.shape[0] != EXPECTED_OBSERVATION_SIZE:
                self.logger.warning(f"[get_observation] LSTM output size mismatch: expected {EXPECTED_OBSERVATION_SIZE}, got {lstm_output.shape[0]}.")
                return None

            self.logger.info(f"[get_observation] Processed observation: {lstm_output}")
            return lstm_output

        except Exception as e:
            self.logger.error(f"[get_observation] Error in get_observation: {e}")
            return None
