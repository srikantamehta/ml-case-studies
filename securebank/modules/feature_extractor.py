import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.pipeline import Pipeline


class Feature_Extractor:

    def __init__(self):
        self.transformed_data = {}
        self.version = "1.0"
        # Define numeric and categorical features
        self.numeric_features = ['amt', 'merch_lat', 'merch_long', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos']
        self.categorical_features = ['category', 'merchant']

    def extract(self, training_dataset_filename: str, testing_dataset_filename: str):
        """
        Loads the training and testing datasets from parquet files.
        """
        # Load the datasets from parquet files
        self.training_dataset = pd.read_parquet(training_dataset_filename)
        self.testing_dataset = pd.read_parquet(testing_dataset_filename)
        
        return self.training_dataset, self.testing_dataset

    def transform(self, training_dataset=None, test_dataset=None, inference_data=None):
        """
        Transforms the datasets into features useful for training the model, including time-based features, 
        numeric scaling, and target encoding. For inference data, feature engineering is applied but scaling and 
        encoding is skipped (left for prediction).
        
        Args:
            training_dataset (pd.DataFrame): The training dataset.
            test_dataset (pd.DataFrame): The test dataset.
            inference_data (pd.DataFrame): A single data point for inference.
        
        Returns:
            Depending on the input type (training, testing, or inference):
            - Transformed training and test datasets with scaling and encoding.
            - Feature engineered but non-scaled, non-encoded inference data.
        """
        if inference_data is not None:
            # Handle the inference case, returning unscaled/unencoded features
            self._add_time_features(inference_data)
            return inference_data  # Return the feature engineered inference data without scaling or encoding

        else:
            # Handle training and test datasets (with labels)
            self._add_time_features(training_dataset)
            if test_dataset is not None:
                self._add_time_features(test_dataset)

            # Scaling the numerical features
            numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
            X_train_numeric = numeric_transformer.fit_transform(training_dataset[self.numeric_features])
            X_test_numeric = numeric_transformer.transform(test_dataset[self.numeric_features]) if test_dataset is not None else None

            # Target encoding for categorical features
            target_encoder = TargetEncoder(random_state=42)
            X_train_encoded = target_encoder.fit_transform(training_dataset[self.categorical_features], training_dataset['is_fraud'])
            X_test_encoded = target_encoder.transform(test_dataset[self.categorical_features]) if test_dataset is not None else None

            # Convert numeric and categorical arrays back into DataFrames
            X_train_numeric = pd.DataFrame(X_train_numeric, columns=self.numeric_features, index=training_dataset.index)
            X_test_numeric = pd.DataFrame(X_test_numeric, columns=self.numeric_features, index=test_dataset.index) if test_dataset is not None else None
            X_train_encoded = pd.DataFrame(X_train_encoded, columns=self.categorical_features, index=training_dataset.index)
            X_test_encoded = pd.DataFrame(X_test_encoded, columns=self.categorical_features, index=test_dataset.index) if test_dataset is not None else None

            # Combine numeric and categorical features
            X_train = pd.concat([X_train_numeric, X_train_encoded], axis=1)
            X_test = pd.concat([X_test_numeric, X_test_encoded], axis=1) if test_dataset is not None else None

            # Store the transformed datasets along with target labels
            self.transformed_data = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': training_dataset['is_fraud'],
                'y_test': test_dataset['is_fraud'] if test_dataset is not None else None
            }

            # Return transformed datasets, scaler, and encoder
            return X_train, X_test, training_dataset['is_fraud'], test_dataset['is_fraud'], numeric_transformer, target_encoder

    def _add_time_features(self, dataset):
        """
        Adds time-based features (hour_of_day, day_of_week, month) and cyclical encoding for those features.
        
        Args:
            dataset (pd.DataFrame): The dataset to which the time-based features will be added.
        """
        dataset['trans_date_trans_time'] = pd.to_datetime(dataset['trans_date_trans_time'])
        dataset['hour_of_day'] = dataset['trans_date_trans_time'].dt.hour
        dataset['day_of_week'] = dataset['trans_date_trans_time'].dt.dayofweek
        dataset['month'] = dataset['trans_date_trans_time'].dt.month

        # Cyclical encoding for hour_of_day, day_of_week, and month
        dataset['hour_sin'] = np.sin(2 * np.pi * dataset['hour_of_day'] / 24)
        dataset['hour_cos'] = np.cos(2 * np.pi * dataset['hour_of_day'] / 24)
        dataset['day_sin'] = np.sin(2 * np.pi * dataset['day_of_week'] / 7)
        dataset['day_cos'] = np.cos(2 * np.pi * dataset['day_of_week'] / 7)
        dataset['month_sin'] = np.sin(2 * np.pi * dataset['month'] / 12)
        dataset['month_cos'] = np.cos(2 * np.pi * dataset['month'] / 12)

    def describe(self, *args, **kwargs):
        """
        Computes and returns the significant quality metrics of the transformed dataset. 
        """
        # Retrieve the transformed data
        X_train = self.transformed_data.get('X_train')
        X_test = self.transformed_data.get('X_test')
        y_train = self.transformed_data.get('y_train')
        y_test = self.transformed_data.get('y_test')

        # Dataset metrics
        dataset_description = {
            'version': self.version,
            'description': {
                'training_set_size': X_train.shape[0],
                'test_set_size': X_test.shape[0],
                'feature_count': X_train.shape[1],
                'training_class_distribution': y_train.value_counts(normalize=True).to_dict(), 
                'test_class_distribution': y_test.value_counts(normalize=True).to_dict(),
                'training_stats': X_train.describe().to_dict(),
                'test_stats': X_test.describe().to_dict(),
            }
        }

        # Return the dataset description
        return dataset_description

