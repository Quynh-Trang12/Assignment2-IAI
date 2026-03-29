import os
import sys
from src.utils.logger_setup import get_colored_logger
import numpy as np
import joblib
from pathlib import Path
from tensorflow.keras.models import load_model  # type: ignore

# Suppress TensorFlow C++ backend spam logs before importing Keras
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

logger = get_colored_logger(__name__)


class TrafficFlowPredictor:
    """
    The Machine Learning Inference Engine.
    Loads the trained neural network and scaler artifacts into RAM to execute
    real-time, mathematically scaled traffic flow predictions.
    """

    def __init__(self, model_type: str = "lstm"):
        """
        Initializes the predictor by resolving artifact paths and loading them into memory.

        Args:
            model_type (str): The architecture to load ('lstm', 'gru', or 'fnn').
        """
        # Dynamically resolve the absolute project root
        script_dir = Path(__file__).resolve().parent
        self.project_root = script_dir.parent.parent

        self.model_path = (
            self.project_root / "models" / "saved_models" / f"{model_type}_model.keras"
        )
        self.scaler_path = self.project_root / "models" / "scalers" / "flow_scaler.save"

        self._verify_artifacts()

        # Load artifacts into memory
        try:
            logger.info("Loading mathematical scaler from: %s", self.scaler_path.name)
            self.scaler = joblib.load(self.scaler_path)

            logger.info(
                "Loading %s neural network from: %s",
                model_type.upper(),
                self.model_path.name,
            )
            self.model = load_model(self.model_path)

            logger.info("Inference Engine initialized successfully.")
        except Exception as e:
            logger.error("Failed to initialize Inference Engine: %s", e)
            sys.exit(1)

    def _verify_artifacts(self) -> None:
        """Strictly verifies that Phase 2 artifacts exist before attempting to load them."""
        if not self.scaler_path.exists():
            logger.error(
                "Scaler artifact missing. Ensure Phase 2 notebook has been executed."
            )
            sys.exit(1)
        if not self.model_path.exists():
            logger.error("Model artifact missing: %s", self.model_path.name)
            sys.exit(1)

    def predict_flow(self, historical_sequence: list) -> float:
        """
        Executes a forward pass through the neural network to predict the next 15-min interval.

        Args:
            historical_sequence (list): The exact sequence of previous flow values (e.g., length 3).

        Returns:
            float: The predicted real-world vehicle count.
        """
        try:
            # 1. Format the data for the scaler (requires a 2D vertical array)
            raw_array = np.array(historical_sequence).reshape(-1, 1)

            # 2. Scale the input values down to the [0.0, 1.0] mathematical range
            scaled_array = self.scaler.transform(raw_array)

            # 3. Format the data for the Recurrent Neural Network (Samples, Time Steps, Features)
            # Example shape for 1 prediction using 3 time steps: (1, 3, 1)
            rnn_input = scaled_array.reshape(1, len(historical_sequence), 1)

            # 4. Execute the mathematical forward pass (verbose=0 disables the loading bar)
            scaled_prediction = self.model.predict(rnn_input, verbose=0)

            # 5. Inverse the scaling back to real-world vehicle counts
            real_prediction = self.scaler.inverse_transform(scaled_prediction)

            # Return the raw float value, clamped to 0 to prevent impossible negative traffic
            return max(0.0, float(real_prediction[0][0]))

        except Exception as e:
            logger.error("Inference failure during forward pass: %s", e)
            # Failsafe: return an exceptionally high gridlock number so the routing
            # algorithm avoids this broken edge rather than crashing.
            return 2000.0


# ---------------------------------------------------------
# Unit Testing / Standalone Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    logger.info("Executing Inference Engine Validation Test...")

    # Instantiate the predictor
    predictor = TrafficFlowPredictor(model_type="lstm")

    # Provide a sample 3-step historical sequence (e.g., 45 minutes of traffic building up)
    sample_history = [120.0, 145.0, 160.0]

    predicted_vehicles = predictor.predict_flow(sample_history)

    logger.info("Input Sequence: %s", sample_history)
    logger.info("Predicted Next Interval: %.2f vehicles", predicted_vehicles)
