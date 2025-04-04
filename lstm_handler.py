import os
import torch
from lstm_model import LSTMModel, LSTMTrainer
import logging
import numpy as np

logger = logging.getLogger(__name__)

class LSTMHandler:
    def __init__(self):
        # Adjusted to match 10-second EEG sequences (flattened feature vector size: 43008)
        input_size = 43008  # Flattened feature vector size for 1 second
        hidden_size = 128  # Hidden layer size
        output_size = 5  # Example: Predicting one of 5 possible actions discrete_action (integer in [0, 8] for action type), cont1, cont2, cont3, cont4
        num_layers = 2

        self.model = LSTMModel(input_size, hidden_size, output_size, num_layers)
        self.trainer = LSTMTrainer(self.model)

        # Load pre-trained model if available
        model_path = os.path.join(os.getcwd(), "models", "lstm_model.pth")
        if os.path.exists(model_path):
            self.trainer.load_model(model_path, logger)
            logger.info("Loaded pre-trained LSTM model.")
        else:
            logger.warning("No pre-trained LSTM model found. Using untrained model.")

    def predict(self, features):
        """
        Make predictions using the LSTM model.
        :param features: EEG feature tensor of shape (10, 14, 256)
        :return: np.ndarray of shape (5,), first is for discrete action, rest are continuous
        """
        try:
            expected_size = 43008  # 43008
            if np.prod(features.shape) != expected_size:
                raise ValueError(f"Expected input size {expected_size}, but got {np.prod(features.shape)}")

            # Reshape to (1, 10, 43008)
            feature_tensor = torch.tensor(features, dtype=torch.float32).reshape(1, 10, -1)
            logger.info(f"Feature tensor shape: {feature_tensor.shape}")

            # Forward pass
            lstm_output = self.model(feature_tensor).detach().numpy().flatten()

            # Clip discrete part, keep continuous as-is
            discrete_action = lstm_output[0]
            continuous_values = lstm_output[1:]

            logger.info(f"LSTM raw output: {lstm_output}")
            return np.concatenate([[discrete_action], continuous_values])

        except (RuntimeError, ValueError) as e:
            logger.error(f"Error during LSTM prediction: {e}")
            return None



    def predict_sequence(self, feature_sequence):
        """
        Predict the action based on a sequence of features.
        :param feature_sequence: Feature sequence of shape (10, 43008).
        :return: Predicted action probabilities.
        """
        try:
            # Validate input shape
            if feature_sequence.shape != (10, 43008):
                raise ValueError(f"Expected input shape (10, 43008), but got {feature_sequence.shape}")

            # Reshape input to (batch_size=1, sequence_length=10, input_size=43008)
            feature_tensor = torch.tensor(feature_sequence, dtype=torch.float32).unsqueeze(0)
            logger.info(f"Feature tensor shape: {feature_tensor.shape}")

            # Run through LSTM model
            lstm_output = self.model(feature_tensor).detach().numpy().flatten()

            # Apply softmax for classification
            lstm_output = np.exp(lstm_output) / np.sum(np.exp(lstm_output))  # Softmax normalization
            logger.info(f"LSTM output probabilities: {lstm_output}")

            # Normalize continuous values to the range [-1, 1]
            continuous_values = lstm_output[1:]  # Exclude the first value (discrete action)
            continuous_values = 2 * (continuous_values - np.min(continuous_values)) / (np.max(continuous_values) - np.min(continuous_values)) - 1

            # Combine discrete and continuous values
            normalized_output = np.concatenate([[lstm_output[0]], continuous_values])
            logger.info(f"Normalized LSTM output: {normalized_output}")

            return normalized_output

        except (RuntimeError, ValueError) as e:
            logger.error(f"Error during LSTM prediction: {e}")
            return None

    def named_parameters(self):
        """
        Expose the named parameters of the underlying LSTM model.
        :return: Named parameters of the model.
        """
        return self.model.named_parameters()
