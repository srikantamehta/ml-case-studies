# Data Pipeline Design

The data pipeline consists of three modules: `Raw_Data_Handler`, `Dataset_Designer`, and `Feature_Extractor`.

---

## 1. Raw Data Handler (`raw_data_handler.py`)

### Purpose:
The `Raw_Data_Handler` class is responsible for extracting, transforming, and cleaning raw data from multiple source datasets (e.g., customers, transactions, and fraud data).

### Design Decisions:

#### **1.1. Extraction Process (`extract` method)**:
- **Data Source Formats**: The data comes in different formats:
  - Customer information (`CSV`)
  - Transaction data (`Parquet`)
  - Fraud information (`JSON`)

To handle the differences, the `extract()` method reads each file into a `pandas.DataFrame`. This provides a uniform representation of the data, regardless of the original format.

#### **1.2. Data Cleaning and Transformation (`transform` method)**:
- **Merging Data**: The data from the three sources is merged on common keys:
  - The fraud data is merged with the transaction data on `trans_num`.
  - The resulting dataset is merged with the customer data on `cc_num` (credit card number).
  
- **Data Standardization**: Columns like `trans_date_trans_time` are converted to a standard datetime format, and string columns (e.g., `category`, `merchant`) are transformed to lowercase to maintain consistency.

- **Handling Missing Values**: The missing values in `category` are imputed based on `merchant` values, and any remaining missing values in `category` and `merchant` are replaced with `'unknown'`.

- **Decision**:
  - **Merge Strategy**: The decision to use inner joins ensures that only records with complete information (i.e., matching transaction and fraud data) are included. This prevents the inclusion of records that do not have corresponding fraud or customer information.
  - **Imputation**: Imputing missing `category` based on `merchant` allows us to preserve data where possible.

#### **1.3. Dataset Description (`describe` method)**:
- This method computes metrics like the percentage of missing values, class distribution, and summary statistics (mean, standard deviation) for numeric columns.
- These metrics provide a good high level overview of the raw dataset and its quality. 

#### **1.4. Storage of Processed Data (`load` method)**:
- The `load()` method saves the processed data in a Parquet format. Parquet is chosen primarily because of its efficiency in handling large datasets.

---

## 2. Dataset Designer (`dataset_design.py`)

### Purpose:
The `Dataset_Designer` class is responsible for partitioning the data into training and testing sets, which will be used for building and evaluating machine learning models.

### Design Decisions:

#### **2.1. Data Extraction (`extract` method)**:
- The raw dataset is loaded from a Parquet file into a pandas dictionary to enable efficient data processing.

#### **2.2. Data Sampling (`sample` method)**:
- The dataset is partitioned into training and test sets using a default **80/20 split**. An option for stratified sampling is provided to ensure that the training and test sets have a similar distribution of the target variable (`is_fraud`). This ensures that the training and test sets maintain the same class balance.

#### **2.3. Dataset Description (`describe` method)**:
- This method provides metrics like the size of the training and test sets, the number of features, and the class distribution for the `is_fraud` column.
- These metrics help monitor data partitioning and ensure that the dataset is properly balanced and split.

#### **2.4. Saving Partitioned Data (`load` method)**:
- The partitioned datasets are saved as Parquet files. This allows the data to be efficiently stored and retrieved later for training.

---

## 3. Feature Extractor (`feature_extractor.py`)

### Purpose:
The `Feature_Extractor` class is responsible for extracting and transforming features from the partitioned datasets into a format suitable for training machine learning models.

### Design Decisions:

#### **3.1. Data Extraction (`extract` method)**:
- The training and test datasets are read from Parquet files.

#### **3.2. Feature Transformation (`transform` method)**:
- **Time-Based Features**: New features such as `hour_of_day`, `day_of_week`, `month`, and `time_since_last_transaction` are created from the `trans_date_trans_time` column. These features help capture the temporal patterns of fraudulent transactions. Initial analysis showed that fraudulent transactions often follow time-based patterns, such as occurring more frequently at specific hours or days, which is why these time features were engineered.
  
- **Categorical Encoding**: The categorical columns `category` and `merchant` are target-encoded based on the target variable `is_fraud`. Target encoding is chosen to handle high-cardinality categorical variables like `merchant` efficiently. 
  
- **Scaling Numerical Features**: Numeric features are scaled using `StandardScaler` to ensure that they have similar ranges, which helps certain models converge faster and avoids effecting models sensitive to feature magnitude.

#### **3.3. Dataset Description (`describe` method)**:
- This method computes key metrics such as the number of features, missing values, and basic statistics like mean and standard deviation for each feature.
- This helps with understanding the dataset's structure and feature distribution and ensures data quality before training models.


