import os
import pandas as pd
from sklearn.model_selection import train_test_split

class Dataset_Designer:
    
    def __init__(self):
        self.raw_dataset = None
        self.partitioned_data = {}
        self.version = "1.0"
        self.storage_path = ""
        self.description = {}

    def extract(self, raw_dataset_filename: str):
        """
        Loads the raw dataset from a parquet file.
        """
        # Read the parquet raw data file
        self.raw_dataset = pd.read_parquet(raw_dataset_filename)
        return self.raw_dataset
    
    def sample(self, raw_dataset, test_size=0.2, random_state=42, stratify=True):
        """
        Splits the raw dataset into training and test sets with an option for stratification.
        """
        # Optional stratificaton based on target column
        stratify_option = raw_dataset['is_fraud'] if stratify else None

        # Split into training and test datasets
        train_set, test_set = train_test_split(raw_dataset, test_size=test_size, random_state=random_state, stratify=stratify_option)

        # Store partitioned data
        self.partitioned_data = {
            'train_set': train_set,
            'test_set': test_set
        }
        return [train_set, test_set]
    
    def describe(self, *args, **kwargs):
        """
        Computes and returns key metrics of the partitioned data, such as class distribution, 
        dataset size, and feature count.
        """
        # Retrieve partitioned data
        train_set = self.partitioned_data.get('train_set', pd.DataFrame())
        test_set = self.partitioned_data.get('test_set', pd.DataFrame())

        # Get the class distribution for 'is_fraud'
        train_class_distribution = train_set['is_fraud'].value_counts(normalize=True).to_dict()
        test_class_distribution = test_set['is_fraud'].value_counts(normalize=True).to_dict()

        dataset_description = {
            'version': self.version,
            'storage': self.storage_path,
            'description': {
                'training_set_size': train_set.shape[0],
                'test_set_size': test_set.shape[0],
                'feature_count': train_set.shape[1] - 1,
                'training_class_distribution': train_class_distribution,
                'test_class_distribution': test_class_distribution
            }
        }

        self.description = dataset_description
        return dataset_description

    def load(self, output_filename: str):
        """
        Saves the partitioned training and test datasets to parquet files.
        """
        # Create output directory
        output_directory = os.path.join(os.getcwd(), 'processed_data')
        os.makedirs(output_directory, exist_ok=True) 
        output_base_path = os.path.join(output_directory, output_filename)

        # Save the partitioned datasets
        self.partitioned_data['train_set'].to_parquet(f"{output_base_path}_train_set.parquet")
        self.partitioned_data['test_set'].to_parquet(f"{output_base_path}_test_set.parquet")

        # Update storage path and print confirmation
        self.storage_path = output_base_path
        print(f"Saved partitioned datasets to {output_base_path}")