import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.pipeline import Pipeline


class Feature_Extractor:

    def __init__(self):
        self.transformed_data = {}
        self.version = "1.0"

    def extract(self, training_dataset_filename: str, testing_dataset_filename: str):
        """
        Loads the training and testing datasets from parquet files.
        """
        # Load the datasets from parquet files
        self.training_dataset = pd.read_parquet(training_dataset_filename)
        self.testing_dataset = pd.read_parquet(testing_dataset_filename)
        
        return self.training_dataset, self.testing_dataset

    def transform(self, training_dataset, test_dataset):
        """
        Transforms the datasets into features useful for training the model, including time-based features, 
        numeric scaling, and target encoding.
        """
        # Separate target column (is_fraud)
        y_train = training_dataset['is_fraud']
        y_test = test_dataset['is_fraud']

        # Convert 'trans_date_trans_time' to datetime format
        training_dataset['trans_date_trans_time'] = pd.to_datetime(training_dataset['trans_date_trans_time'])
        test_dataset['trans_date_trans_time'] = pd.to_datetime(test_dataset['trans_date_trans_time'])
        
        # Add time-based features to the training and testing datasets
        for dataset in [training_dataset, test_dataset]:
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

        # Define numerical and categorical columns
        numeric_features = ['amt', 'merch_lat', 'merch_long',
                            'hour_sin','hour_cos','day_sin','day_cos','month_sin','month_cos']
        categorical_features = ['category', 'merchant']

        # Scaling the numerical features (including 'time_since_last_transaction')
        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

        # Apply the numeric transformation to the training and testing datasets
        X_train_numeric = numeric_transformer.fit_transform(training_dataset[numeric_features])
        X_test_numeric = numeric_transformer.transform(test_dataset[numeric_features])

        # Target Encoder for categorical features
        target_encoder = TargetEncoder()

        # Apply target encoding to the training and testing datasets
        X_train_encoded = target_encoder.fit_transform(training_dataset[categorical_features], training_dataset['is_fraud'])
        X_test_encoded = target_encoder.transform(test_dataset[categorical_features])

        # Convert the encoded arrays back into DataFrames
        X_train_encoded = pd.DataFrame(X_train_encoded, columns=categorical_features, index=training_dataset.index)
        X_test_encoded = pd.DataFrame(X_test_encoded, columns=categorical_features, index=test_dataset.index)

        # Convert numeric arrays back into DataFrames
        X_train_numeric = pd.DataFrame(X_train_numeric, columns=numeric_features, index=training_dataset.index)
        X_test_numeric = pd.DataFrame(X_test_numeric, columns=numeric_features, index=test_dataset.index)

        # Combine encoded categorical and scaled numerical features
        X_train = pd.concat([X_train_numeric, X_train_encoded], axis=1)
        X_test = pd.concat([X_test_numeric, X_test_encoded], axis=1)

        # Store the transformed datasets along with target labels
        self.transformed_data = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }

        return X_train, X_test, y_train, y_test
    
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

