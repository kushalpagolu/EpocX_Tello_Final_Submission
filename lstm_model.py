import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from datetime import datetime

# Ensure the logs directory exists
logs_dir = "/Users/kushalpagolu/Desktop/EmotivEpoch/epoch_tello_RL_3DBrain/logs"
os.makedirs(logs_dir, exist_ok=True)

# Generate a timestamped log file name
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f"{logs_dir}/lstm_model_{timestamp}.log"

# Update logging configuration to save logs to a new file for each run
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),  # Save logs to a timestamped file
        logging.StreamHandler()  # Also display logs in the console
    ]
)


class LSTMModel(nn.Module):
    """
    PyTorch-based LSTM model for EEG data.
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass of the LSTM model.
        :param x: Input tensor of shape (batch_size, sequence_length, input_size).
        :return: Output tensor of shape (batch_size, output_size).
        """
        # LSTM calculations
        out, _ = self.lstm(x)  # out shape: (batch_size, sequence_length, hidden_size)
        out = self.fc(out[:, -1, :])  # Fully connected layer on the last time step
        return out


class LSTMTrainer:
    """
    Handles training, saving, and loading of the LSTM model.
    """
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def train(self, training_data, training_labels, epochs=10, batch_size=32, logger=None):
        """
        Trains the LSTM model with the given training data and labels.
        :param training_data: Numpy array of shape (num_samples, time_steps, feature_vector_size).
        :param training_labels: Numpy array of shape (num_samples, action_space_size).
        :param epochs: Number of training epochs.
        :param batch_size: Batch size for training.
        :param logger: Logger instance for logging training progress.
        """
        self.model.train()
        training_data = torch.tensor(training_data, dtype=torch.float32)
        training_labels = torch.tensor(training_labels, dtype=torch.float32)

        for epoch in range(epochs):
            permutation = torch.randperm(training_data.size(0))
            epoch_loss = 0.0

            for i in range(0, training_data.size(0), batch_size):
                indices = permutation[i:i + batch_size]
                batch_data = training_data[indices]
                batch_labels = training_labels[indices]

                self.optimizer.zero_grad()
                outputs = self.model(batch_data)
                loss = self.criterion(outputs, batch_labels)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            if logger:
                logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

    def save_model(self, filepath, logger=None):
        """
        Saves the trained LSTM model to the specified file.
        :param filepath: Path to save the model.
        :param logger: Logger instance for logging.
        """
        try:
            torch.save(self.model.state_dict(), filepath)
            if logger:
                logger.info(f"LSTM model saved to {filepath}.")
        except Exception as e:
            if logger:
                logger.error(f"Error saving LSTM model: {e}")

    def load_model(self, filepath, logger=None):
        """
        Loads a trained LSTM model from the specified file.
        :param filepath: Path to load the model from.
        :param logger: Logger instance for logging.
        """
        try:
            self.model.load_state_dict(torch.load(filepath))
            self.model.eval()
            if logger:
                logger.info(f"LSTM model loaded from {filepath}.")
        except Exception as e:
            if logger:
                logger.error(f"Error loading LSTM model: {e}")

    def save_feature_vector(self, feature_vector, save_path, logger=None):
        """
        Saves the feature vector to a file and logs its shape and size.
        :param feature_vector: The feature vector to save.
        :param save_path: Path to save the feature vector.
        :param logger: Logger instance for logging.
        """
        try:
            np.save(save_path, feature_vector)
            if logger:
                logger.info(f"Feature vector saved to {save_path}.")
                logger.info(f"Feature vector shape: {feature_vector.shape}, size: {feature_vector.size}")
        except Exception as e:
            if logger:
                logger.error(f"Error saving feature vector: {e}")


# Example main method to demonstrate usage
if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Initialize the LSTM model
    input_size = 43008  # Flattened feature vector size for 1 second
    hidden_size = 128  # Example hidden size
    output_size = 6  # Example: Predicting one of 6 possible actions
    model = LSTMModel(input_size, hidden_size, output_size)

    # Example feature sequence
    feature_sequence = np.random.rand(10, input_size)  # Example 10-second feature sequence
    feature_tensor = torch.tensor(feature_sequence, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

    # Run the model
    prediction = model(feature_tensor)
    logger.info(f"Prediction shape: {prediction.shape}, Prediction: {prediction}")
