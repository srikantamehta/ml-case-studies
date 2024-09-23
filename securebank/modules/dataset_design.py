import os
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit

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
        try:
            self.raw_dataset = pd.read_parquet(raw_dataset_filename)
        except FileNotFoundError:
            print(f"Error: {raw_dataset_filename} not found")

        return self.raw_dataset
    
    def sample(self, raw_dataset, test_size=0.2, start_date=None, end_date=None):
        """
        Splits the raw dataset into training and test sets based on transaction time, ensuring that 
        all transactions in the training set occur before those in the test set. Optionally filters the dataset by a date range.

        Args:
            raw_dataset (pd.DataFrame): The raw dataset to sample from.
            test_size (float): Proportion of the dataset to use as the test set (default is 0.2).
            start_date (str): Optional start date to filter the data. Format: 'YYYY-MM-DD'.
            end_date (str): Optional end date to filter the data. Format: 'YYYY-MM-DD'.

        Returns:
            [pd.DataFrame, pd.DataFrame]: The training and test datasets.
        """
        # Sort the dataset by transaction time
        self.raw_dataset = raw_dataset.sort_values(by='trans_date_trans_time').reset_index(drop=True)

        if start_date or end_date:
            if start_date:
                try:
                    raw_dataset = raw_dataset[raw_dataset['trans_date_trans_time'] >= pd.to_datetime(start_date)]
                except ValueError:
                    print(f"Invalid start_date: {start_date}. Please use 'YYYY-MM-DD' format.")
            if end_date:
                try:
                    raw_dataset = raw_dataset[raw_dataset['trans_date_trans_time'] <= pd.to_datetime(end_date)]
                except ValueError:
                    print(f"Invalid end_date: {end_date}. Please use 'YYYY-MM-DD' format.")

        # Determine the index at which to split the dataset
        split_index = int(len(raw_dataset) * (1 - test_size))

        # Split into training and test sets based on the sorted transaction time
        train_set = raw_dataset.iloc[:split_index]
        test_set = raw_dataset.iloc[split_index:]

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

    def load(self, output_filename: str = None):
        """
        Saves the processed raw dataset (self.raw_data) to a parquet file.
        If no output_filename is provided, saves to the default directory: /storage/train_data/.
        """
        # Default output directory if no filename is provided
        if output_filename is None:
            base_dir = os.path.dirname(os.path.dirname(__file__))  # Get one level above the current script directory
            output_directory = os.path.join(base_dir, 'storage', 'train_data')
            os.makedirs(output_directory, exist_ok=True)
            output_base_path = os.path.join(output_directory, 'processed_raw_data')
        else:
            output_directory = os.path.dirname(output_filename)
            os.makedirs(output_directory, exist_ok=True)
            output_base_path = output_filename

        # Save the raw dataset as a parquet file
        self.raw_dataset.to_parquet(f"{output_base_path}.parquet")

        # Update storage path and print confirmation
        self.storage_path = output_base_path
        print(f"Saved raw dataset to {output_base_path}.parquet")