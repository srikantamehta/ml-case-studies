import joblib
from datetime import datetime
import os
import pandas as pd
from modules.feature_extractor import Feature_Extractor

class Pipeline:

    def __init__(self, version: str = 'random_forest_model_1'):
        """
        Initializes the inference pipeline object and loads the default model, scaler, and encoder.
        """
        self.version = version
        self.model = None
        self.scaler = None
        self.target_encoder = None
        self.feature_extractor = Feature_Extractor()  # Initialize feature extractor for transformations
        self.history = []

        self.select_model(version)

    def list_models(self, model_dir: str=None):
        """
        Lists all available models in the specified directory or in the default storage/models/artifacts/ directory.
        
        Args:
            model_dir (str): Optional. The directory to look for models. If not provided, it defaults to storage/models/artifacts/.
        
        Returns:
            List[str]: A list of available model versions.
        """
        base_model_dir = os.path.abspath('storage/models/artifacts/')
        
        # Check if the directory exists
        if not os.path.exists(base_model_dir):
            raise FileNotFoundError(f"Model directory not found: {base_model_dir}")
        
        # List directories or model versions
        model_versions = [d for d in os.listdir(base_model_dir) if os.path.isdir(os.path.join(base_model_dir, d))]
        
        if not model_versions:
            raise FileNotFoundError("No models available in the model directory.")
        
        return model_versions

    def select_model(self, version: str):
        """
        Loads a model, scaler, and target encoder from a catalog of pre-trained models in storage/models/artifacts/.
        """
        base_model_dir = os.path.abspath('storage/models/artifacts/')
        model_path = os.path.join(base_model_dir, version, 'model.pkl')
        
        if os.path.exists(model_path):
            model_package = joblib.load(model_path)
            self.model = model_package['model']
            self.scaler = model_package['numeric_transformer']
            self.target_encoder = model_package['target_encoder']
            self.threshold = model_package['threshold']
            self.model_stats = model_package['model_stats']

            if not self.model:
                raise ValueError(f"Model not found in {model_path}")
            if not self.scaler:
                raise ValueError(f"Numeric transformer not found in {model_path}")
            if not self.target_encoder:
                raise ValueError(f"Target encoder not found in {model_path}")
            if not self.threshold:
                raise ValueError(f"Model Threshold value not found in {model_path}")
            if not self.model_stats:
                raise ValueError(f"Model Stats not found in {model_path}")
        
            print(f"Model, scaler, and target encoder loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found in {model_path}")
        
        self.version = version
        
    def predict(self, input_data: dict):
        """
        Uses the specified model to perform predictions on the input data.
        """
        # If a single data point is provided, convert it into a list
        if isinstance(input_data, dict):
            input_data = [input_data]
        
        input_df = pd.DataFrame(input_data)

        # Convert and preprocess the input data using feature extractor
        transformed_data = self.feature_extractor.transform(inference_data=input_df)

        # Extract numeric and categorical features
        numeric_features = self.feature_extractor.numeric_features
        categorical_features = self.feature_extractor.categorical_features

        # Scale and encode the data
        X_numeric = pd.DataFrame(self.scaler.transform(transformed_data[numeric_features]), 
                                 columns=numeric_features)
        X_encoded = pd.DataFrame(self.target_encoder.transform(transformed_data[categorical_features]), 
                                 columns=categorical_features)

        # Combine numeric and categorical features
        X_input = pd.concat([X_numeric, X_encoded], axis=1)
        
        # Predict probablilties using the model and threshold
        proba_predictions = self.model.predict_proba(X_input)[:,1]

        # Generate predicted classifications
        predictions = proba_predictions >= self.threshold

        # Obtain Prediction time
        prediction_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Store the predictions in history for each data point
        for i, data_point in enumerate(input_data):
            self.history.append({
                "input": data_point, 
                "model": self.version,
                "prediction": bool(predictions[i]), 
                "probability": proba_predictions[i],
                "prediction_time": prediction_time
            })

        # Return a single prediction if one data point was passed, else return a list
        return bool(predictions[0]) if len(input_data) == 1 else [bool(prediction) for prediction in predictions]

    def get_history(self):
        """
        Returns information on previous system predictions.
        
        Returns:
            history (List[Dict]): A list containing the input data and corresponding predictions.
        """
        return self.history
    
    def get_model_stats(self):
        """
        Returns the model stats from training.
        """
        return self.model_stats
