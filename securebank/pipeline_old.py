import joblib
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
            print(f"Model, scaler, and target encoder loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found in {model_path}")
        
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
        
        # Predict using the model
        predictions = self.model.predict(X_input)

        # Store the predictions in history for each data point
        for i, data_point in enumerate(input_data):
            self.history.append({
                "input": data_point, 
                "prediction": bool(predictions[i])  # Convert each prediction to a boolean
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
