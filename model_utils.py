import os
import logging
from stable_baselines3 import PPO

MODEL_FILENAME = os.path.join(os.getcwd(), "models", "drone_rl_eeg_human_loop")

def load_or_create_model(env):
    logger = logging.getLogger(__name__)
    if os.path.exists(f"{MODEL_FILENAME}.zip"):
        try:
            model = PPO.load(MODEL_FILENAME, env=env)
            logger.info("Loaded existing model")
        except Exception as e:
            logger.error(f"Error loading model: {e}. Creating a new model.")
            model = PPO("MlpPolicy", env, verbose=1)
    else:
        model = PPO("MlpPolicy", env, verbose=1)
        logger.info("Created new model")
    return model

def log_model_parameters(model):
    logger = logging.getLogger(__name__)
    if model is not None:
        logger.info("Logging model parameters:")
        for name, param in model.policy.named_parameters():
            logger.info(f"Parameter {name}: {param.data}")

def save_model_parameters(model, filename="model_parameters.txt"):
    if model is not None:
        with open(filename, "w") as f:
            for name, param in model.policy.named_parameters():
                f.write(f"Parameter {name}: {param.data}\n")
        logging.info(f"Model parameters saved to {filename}")
