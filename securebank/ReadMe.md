# Fraud Detection System - Setup Guide

The system consists of a **Flask API** running inside a Docker container and a **Streamlit app** running locally. Follow the steps below to get everything up and running.

## Prerequisites
Before you begin, ensure that you have the following installed:
- Docker: [Install Docker](https://docs.docker.com/get-docker/)
- Docker Compose: [Install Docker Compose](https://docs.docker.com/compose/install/)
- Python 3.x: [Install Python](https://www.python.org/downloads/)
- Streamlit: Install via `pip install streamlit`

## Steps to Run the Application

### 1. Clone the Repository
First, clone this repository to your local machine:

### 2. Cd to the `securebank/` directory
```bash
cd securebank/
```

### 3. Run the docker-setup.py file to build and start the docker container and flask app.
```bash
python docker-setup.py
```
### 4. In another terminal start the streamlit app
```bash
streamlit run streamlit_app.py
```
### 5. A window should open up in your default browser with the streamlit app loaded. 


## Directory Structure:

securebank/
│
├── analysis/
│   └── data_analysis.ipynb       # Jupyter notebook for data analysis
│   └── model_performance.ipynb  # Jupyter notebook for model training
│
├── data_sources/
│   ├── customer_release.csv      # Customer data source
│   ├── fraud_release.json        # Fraud labels source
│   └── transactions_release.parquet # Transaction data source
│
├── modules/
│   ├── dataset_design.py         # Dataset design module
│   ├── feature_extractor.py      # Feature extraction module
│   ├── raw_data_handler.py       # Raw data handling module     
│
├── storage/
│   └── models/
│       └── artifacts/            # Folder containing saved models
│           ├── bagging_model_1/  
│           ├── extratrees_model_1/
│           └── random_forest_model_1/
│
├── app.py                        # Streamlit app main file
├── docker-compose.yml            # Docker Compose configuration
├── docker-setup.py               # Script to run Docker setup
├── Dockerfile                    # Dockerfile to build Flask API container
├── pipeline.py                   # Flask API pipeline implementation
├── ReadMe.md                     # Setup and instructions
├── requirements.txt              # Python dependencies
├── System_Report.md              # Project system report
├── test.json                     # Test JSON file with single datapoint for prediction
├── 100_transactions.json         # Test JSON file 100 datapoints for predictions (50 fraud/50 non-fraud)
└── .gitignore                    # Git ignore file
