import os
import pandas as pd

class Raw_Data_Handler:

    def __init__(self):
        # Stores relevant state information
        self.customer_information = None
        self.transaction_information = None
        self.fraud_information = None
        self.raw_data = None

        self.version = "1.0"
        self.storage_path = ""
        self.description = {}

    def extract(self, customer_information_filename: str, transaction_filename: str, fraud_information_filename: str):
        """
        Loads customer, transaction, and fraud information from their respective file formats 
        (CSV, Parquet, JSON).
        """
        # Create dataframes
        try:
            self.customer_information = pd.read_csv(customer_information_filename)
        except FileNotFoundError:
            print(f"Error: {customer_information_filename} not found.")
        try:
            self.fraud_information = pd.read_json(fraud_information_filename, typ='series').reset_index()
        except FileNotFoundError:
            print(f"Error: {fraud_information_filename} not found.")
        try:
            self.transaction_information = pd.read_parquet(transaction_filename).reset_index()
        except FileNotFoundError:
            print(f"Error: {transaction_filename} not found.")
        
        return self.customer_information, self.transaction_information, self.fraud_information

    def transform(self, customer_information, transaction_information, fraud_information):
        """
        Cleans, merges, and processes the raw data from customer, transaction, and fraud sources.
        Handle missing values, column formatting, and initial feature engineering.
        """

        # Add column headers for fraud information dataframe
        fraud_information.columns = ['trans_num','is_fraud']

        # Merge Fraud and Transaction dataframes by trans_num
        transactions_fraud = pd.merge(transaction_information, fraud_information, on='trans_num', how='inner')
        
        # Merge with customer informaiton dataframe by cc_num
        self.raw_data = pd.merge(transactions_fraud,customer_information,on='cc_num',how='inner')
        
        # Convert trans_date_trans_time to datetime
        self.raw_data['trans_date_trans_time'] = pd.to_datetime(self.raw_data['trans_date_trans_time'])
        self.raw_data = self.raw_data.sort_values('trans_date_trans_time').reset_index(drop=True)
        
        # Convert all string columns to lowercase
        string_columns = self.raw_data.select_dtypes(include='object').columns
        self.raw_data[string_columns] = self.raw_data[string_columns].apply(lambda col: col.str.lower())

        # Drop all rows with missing is_fraud value
        self.raw_data = self.raw_data.dropna(subset=['is_fraud'])

        # Impute missing unix_time values from transaction time
        self.raw_data['unix_time'] = self.raw_data.apply(
            lambda row: row['trans_date_trans_time'].timestamp() if pd.isnull(row['unix_time']) else row['unix_time'], axis=1
        )

        # Impute missing 'category' values based on 'merchant'
        self.raw_data['category'] = self.raw_data['category'].fillna(
            self.raw_data.groupby('merchant')['category'].transform(lambda x: x.mode()[0] if not x.mode().empty else None)
        )

        # Impute any remaining missing 'category' and 'merchant' values with 'unknown'
        self.raw_data['category'] = self.raw_data['category'].fillna('unknown')
        self.raw_data['merchant'] = self.raw_data['merchant'].fillna('unknown')

        # Fix column order
        column_order = [
            'trans_date_trans_time', 'cc_num', 'merchant', 'category', 'amt', 'first', 'last', 
            'sex', 'street', 'city', 'state', 'zip', 'lat', 'long', 'city_pop', 'job', 
            'dob', 'trans_num', 'unix_time', 'merch_lat', 'merch_long', 'is_fraud'
        ]
        self.raw_data = self.raw_data[[col for col in column_order if col in self.raw_data.columns]]

        return self.raw_data


    def describe(self, include_z_score=False, z_score_threshold=3):
        """
        Computes quality metrics for the transformed dataset, including completeness, 
        summary statistics, and optional z-score outlier analysis.
        """
        # Completeness metrics
        table_completeness = (self.raw_data.notnull().sum().sum()) / self.raw_data.size
        row_completeness = (self.raw_data.notnull().all(axis=1).sum()) / self.raw_data.shape[0]
        column_completeness = (self.raw_data.notnull().all(axis=0).sum()) / self.raw_data.shape[1]

        # Stats summary
        stats_summary = {}
        
        # Compute null count for each column
        for col in self.raw_data.columns: 
            stats_summary[col] = {
                'null_count': self.raw_data[col].isnull().sum()
            }

        # Select numeric columns
        numeric_columns = self.raw_data.select_dtypes(include=['float64', 'int64']).columns

        for col in numeric_columns:

            # Calculate numeric column metrics mean and std
            mean_value = self.raw_data[col].mean()
            std_value = self.raw_data[col].std()
            

            # Add metrics to summary
            stats_summary[col].update({
                'mean': mean_value,
                'std': std_value
            })

            # Include optional z-score outlier count
            if include_z_score:
                z_scores = (self.raw_data[col] - mean_value) / self.raw_data[col].std()
                outliers_count = len(z_scores[abs(z_scores) > z_score_threshold])
                stats_summary[col]['outliers_count'] = outliers_count

        # Create quality metrics dictionary
        quality_metrics = {
            'row_count': self.raw_data.shape[0],
            'column_count': self.raw_data.shape[1],
            'table_completeness': table_completeness,
            'row_completeness': row_completeness,
            'column_completeness': column_completeness,
            'num_fraudulent_transactions': self.raw_data['is_fraud'].sum(),
            'unique_customers': self.raw_data['cc_num'].nunique(),
            'stats_summary': stats_summary
        }

        # Final dictionary
        self.description = {
            'version': '1.0',
            'storage': self.storage_path,
            'description': quality_metrics
        }

        return self.description
    
    def load(self, output_filename: str):
        """
        Saves the transformed dataset to a specified output filename in Parquet format.
        Ensures that the output directory exists before saving.
        """
        # Define a default output directory relative to the current working directory
        output_directory = os.path.join(os.getcwd(), 'processed_data')
        # Make sure the output directory exists
        os.makedirs(output_directory, exist_ok=True)
        # Full path
        output_base_path = os.path.join(output_directory, output_filename)
        # Save File
        self.raw_data.to_parquet(f"{output_base_path}.parquet")
        # Update the storage path
        self.storage_path = output_base_path
        # Print confirmation 
        print(f"Saved to {output_base_path}.parquet")