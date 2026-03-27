from typing import Dict, Tuple, Optional, Any
import json
import os

import joblib
import numpy as np
try:
    from tensorflow.keras.models import load_model
except ImportError:
    load_model = None

from sequence_store import SequenceStore


class FlowPredictor:
    """
    Predicts traffic flow for a graph edge using one of four modes:

    - mock: fixed hardcoded values for quick testing
    - baseline: average flow per SCATS site, with optional hour adjustment
    - lstm: sequence-based prediction using the trained LSTM model
    - gru: sequence-based prediction using the trained GRU model

    Important:
    The LSTM and GRU models were trained on sequences of shape (4, 1):
        [flow_t-4, flow_t-3, flow_t-2, flow_t-1] -> predict flow_t

    Therefore, this class must build a real 4-step flow sequence for the
    mapped SCATS site before calling the deep learning models.
    """

    SUPPORTED_MODELS = {"mock", "baseline", "lstm", "gru", "rf"}

    def __init__(
        self,
        model_type: str = "mock",
        sequence_store: Optional[SequenceStore] = None,
        default_time_index: int = 100,
    ) -> None:
        self.model_type = model_type.lower()

        if self.model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model_type '{model_type}'. "
                f"Supported: {sorted(self.SUPPORTED_MODELS)}"
            )

        # Fallback value if prediction cannot be made
        self.default_flow: float = 100.0

        # Node to SCATS site mapping (lazily populated to ensure reproducibility)
        self.node_to_site_map: Dict[int, str] = {}
        

        # Baseline data
        self.baseline_flow_by_site = self._load_baseline_flows()

        # ML artifacts
        self.lstm_model = None
        self.gru_model = None
        self.rf_model = None
        self.x_scaler = None
        self.y_scaler = None

        if self.model_type in {"lstm", "gru", "rf"}:
            self._load_ml_artifacts()

        # Sequence source for LSTM/GRU
        # Must provide real consecutive flow values per SCATS site.
        self.sequence_store = sequence_store

        # Fixed time index for now.
        # This is the position used to extract the last 4 observations.
        self.default_time_index = default_time_index

    def _load_ml_artifacts(self) -> None:
        """
        Load scalers and trained neural network models from disk.
        """
        base_dir = os.path.dirname(__file__)

        x_scaler_path = os.path.join(base_dir, "x_scaler.save")
        y_scaler_path = os.path.join(base_dir, "y_scaler.save")
        lstm_path = os.path.join(base_dir, "lstm_model.h5")
        gru_path = os.path.join(base_dir, "gru_model.h5")
        rf_path = os.path.join(base_dir, "rf_model.pkl")

        if os.path.exists(x_scaler_path):
            self.x_scaler = joblib.load(x_scaler_path)

        if os.path.exists(y_scaler_path):
            self.y_scaler = joblib.load(y_scaler_path)

        if self.model_type == "lstm":
            if os.path.exists(lstm_path):
                self.lstm_model = load_model(lstm_path, compile=False)

        if self.model_type == "gru":
            if os.path.exists(gru_path):
                self.gru_model = load_model(gru_path, compile=False)

        if self.model_type == "rf":
            if os.path.exists(rf_path):
                self.rf_model = joblib.load(rf_path)

    def _load_baseline_flows(self) -> Dict[str, float]:
        """
        Load average baseline flow per SCATS site from JSON.
        """
        filepath = os.path.join(os.path.dirname(__file__), "baseline_avg_flow.json")

        if not os.path.exists(filepath):
            print(f"[warning] {filepath} not found. Using empty baseline flow map.")
            return {}

        with open(filepath, "r") as f:
            data = json.load(f)

        return {str(k): float(v) for k, v in data.items()}

    def _map_edge_to_site(self, u: int, v: int) -> Optional[str]:
        """
        Map a graph edge (u, v) to a SCATS site ID.
        Returns the SCATS site of the destination node v.
        """
        if v in self.node_to_site_map:
            return self.node_to_site_map[v]
            
        if not hasattr(self, 'baseline_flow_by_site') or not self.baseline_flow_by_site:
            return None
            
        # Extract deterministically sorted list of valid sites
        sorted_sites = sorted(self.baseline_flow_by_site.keys())
        
        # Consistently map numeric node ID to the same SCATS site
        site_id = sorted_sites[v % len(sorted_sites)]
        self.node_to_site_map[v] = site_id
        
        return site_id

    def predict_flow(
        self,
        u: int,
        v: int,
        time_context: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Main prediction entry point.
        Dispatches to the selected model type.
        """
        if self.model_type == "mock":
            flow = self._predict_mock(u, v)

        elif self.model_type == "baseline":
            flow = self._predict_baseline(u, v, time_context)

        elif self.model_type == "lstm":
            flow = self._predict_lstm(u, v, time_context)

        elif self.model_type == "gru":
            flow = self._predict_gru(u, v, time_context)

        elif self.model_type == "rf":
            flow = self._predict_rf(u, v, time_context)

        else:
            flow = self.default_flow

        return max(0.0, float(flow))

    def _predict_mock(self, u: int, v: int) -> float:
        """
        Return a fixed mock flow for testing.
        All flows are derived from mapping the destination node to ensure
        no edge skips the proper mapping layer.
        """
        site_id = self._map_edge_to_site(u, v)
        if site_id is None:
            return self.default_flow
            
        # Return a simple mock value based on the site ID ensuring consistency
        return 1000.0 + (int(site_id) % 500)

    def _predict_baseline(
        self,
        u: int,
        v: int,
        time_context: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Predict flow using average baseline per SCATS site.
        Optional hour-based multiplier is applied to simulate time variation.
        """
        site_id = self._map_edge_to_site(u, v)
        if site_id is None:
            return self.default_flow

        base_flow = self.baseline_flow_by_site.get(site_id, self.default_flow)

        if time_context is None:
            return base_flow

        hour = time_context.get("hour")
        if hour is None:
            return base_flow

        # Simple heuristic adjustment
        if 7 <= hour <= 9:
            return base_flow * 1.20
        if 16 <= hour <= 18:
            return base_flow * 1.15
        if 0 <= hour <= 5:
            return base_flow * 0.80

        return base_flow

    def _get_sequence_for_edge(
        self,
        u: int,
        v: int,
        time_context: Optional[Dict[str, Any]] = None
    ) -> Optional[np.ndarray]:
        """
        Build the exact type of sequence used during training:

            4 consecutive flow values from one SCATS site

        Returns:
            numpy array of shape (1, 4, 1), ready for model input
            OR None if the sequence cannot be built.
        """
        if self.sequence_store is None:
            return None

        site_id = self._map_edge_to_site(u, v)
        if site_id is None:
            return None

        # Use a fixed time index unless caller provides one
        t = self.default_time_index
        if time_context is not None and "t" in time_context:
            t = time_context["t"]

        # Expecting a list like [382, 347, 369, 300]
        seq = self.sequence_store.get_sequence(site_id, t)
        if seq is None or len(seq) != 4:
            return None

        # Scale exactly the same way as training:
        # scaler was fit on values reshaped to (-1, 1)
        seq = np.array(seq, dtype=np.float32).reshape(-1, 1)

        if self.x_scaler is None:
            return None

        seq_scaled = self.x_scaler.transform(seq)

        # Final shape for LSTM/GRU: (batch, timesteps, features) = (1, 4, 1)
        return seq_scaled.reshape(1, 4, 1)

    def _predict_lstm(
        self,
        u: int,
        v: int,
        time_context: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Predict next flow using the trained LSTM model.
        """
        if self.lstm_model is None or self.x_scaler is None or self.y_scaler is None:
            return self.default_flow

        X_seq = self._get_sequence_for_edge(u, v, time_context)
        if X_seq is None:
            return self.default_flow

        y_pred_scaled = self.lstm_model.predict(X_seq, verbose=0)

        # Ensure 2D shape before inverse_transform
        if len(y_pred_scaled.shape) == 1:
            y_pred_scaled = y_pred_scaled.reshape(-1, 1)

        flow = self.y_scaler.inverse_transform(y_pred_scaled)[0][0]
        return float(flow)

    def _predict_gru(
        self,
        u: int,
        v: int,
        time_context: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Predict next flow using the trained GRU model.
        """
        if self.gru_model is None or self.x_scaler is None or self.y_scaler is None:
            return self.default_flow

        X_seq = self._get_sequence_for_edge(u, v, time_context)
        if X_seq is None:
            return self.default_flow

        y_pred_scaled = self.gru_model.predict(X_seq, verbose=0)

        # Ensure 2D shape before inverse_transform
        if len(y_pred_scaled.shape) == 1:
            y_pred_scaled = y_pred_scaled.reshape(-1, 1)

        flow = self.y_scaler.inverse_transform(y_pred_scaled)[0][0]
        return float(flow)

    def _predict_rf(
        self,
        u: int,
        v: int,
        time_context: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Predict next flow using the trained Random Forest model.
        """
        if self.rf_model is None or self.x_scaler is None or self.y_scaler is None:
            return self.default_flow

        X_seq = self._get_sequence_for_edge(u, v, time_context)
        if X_seq is None:
            return self.default_flow

        # RF expects 2D array: (1, 4)
        X_seq_2d = X_seq.reshape(1, -1)
        y_pred_scaled = self.rf_model.predict(X_seq_2d)

        # Ensure 2D shape before inverse_transform
        if len(y_pred_scaled.shape) == 1:
            y_pred_scaled = y_pred_scaled.reshape(-1, 1)

        flow = self.y_scaler.inverse_transform(y_pred_scaled)[0][0]
        return float(flow)