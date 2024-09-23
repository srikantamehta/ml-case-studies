import streamlit as st
import requests
import json
import os
from modules.dataset_design import Dataset_Designer
from modules.raw_data_handler import Raw_Data_Handler
import pandas as pd

# Flask API base URL
FLASK_API_URL = "http://localhost:5001" 

# Main UI
st.title("Fraud Detection System")

# Create Tabs 
tabs = st.tabs(["Fraud Detection", "Data Processing"])

with tabs[0]:

    st.header("Fraud Detection")

    # Fetch available models from the Flask API
    try:
        response = requests.get(f"{FLASK_API_URL}/list_models/")
        if response.status_code == 200:
            models = response.json().get('available_models', [])
        else:
            st.error(f"Error: Failed to fetch models. Status code: {response.status_code}")
            models = []
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")
        models = []

    # Dropdown to select a model
    if models:
        selected_model = st.selectbox("Select a model", models)
        
        # Allow users to select a model for predictions
        if st.button("Select Model"):
            try:
                # Send a POST request to select the chosen model
                model_response = requests.post(f"{FLASK_API_URL}/select_model/", json={"model_version": selected_model})
                if model_response.status_code == 200:
                    st.success(f"Model '{selected_model}' selected successfully.")
                else:
                    st.error(f"Error: Failed to select model. Status code: {model_response.status_code}")
            except requests.exceptions.RequestException as e:
                st.error(f"Request failed: {e}")
    else:
        st.write("No models available to select.")

    # Fetch and display the model stats for the selected model
    if st.button("View Model Stats"):
        try:
            response = requests.get(f"{FLASK_API_URL}/model_stats/")
            if response.status_code == 200:
                model_stats = response.json().get("model_stats", {})
                if model_stats:
                    st.write("Model Training Metrics:")
                    st.write(f"F1 Score: {model_stats.get('f1_score', 'N/A')}")
                    st.write(f"Precision: {model_stats.get('precision', 'N/A')}")
                    st.write(f"Recall: {model_stats.get('recall', 'N/A')}")
                else:
                    st.write("No model stats available.")
            else:
                st.error(f"Error: Received status code {response.status_code}")
                st.write(f"Response content: {response.content.decode('utf-8')}")
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")

    st.write("Enter transaction details for fraud prediction or upload a JSON file:")

    # Allow users to choose between manual input or file upload
    input_method = st.radio("Select Input Method", ("Manual Input", "Upload JSON"))

    # If the user selects manual input, display the form fields
    if input_method == "Manual Input":
        # Input form for transaction details
        trans_date_trans_time = st.text_input("Transaction Date & Time (YYYY-MM-DD HH:MM:SS)", "2019-12-01 22:59:12")
        cc_num = st.text_input("Credit Card Number", "371009169330125")
        unix_time = st.text_input("Unix Time", "1575241152.0")
        merchant = st.text_input("Merchant", "fraud_huel-langworth")
        category = st.text_input("Category", "misc_net")
        amt = st.text_input("Amount", "777.68")
        merch_lat = st.text_input("Merchant Latitude", "30.955195")
        merch_long = st.text_input("Merchant Longitude", "-82.705954")

        # Button to make a prediction
        if st.button("Submit Prediction"):
            # Prepare input data
            input_data = {
                "trans_date_trans_time": trans_date_trans_time,
                "cc_num": cc_num,
                "unix_time": float(unix_time),
                "merchant": merchant,
                "category": category,
                "amt": float(amt),
                "merch_lat": float(merch_lat),
                "merch_long": float(merch_long)
            }

            # Send a POST request to Flask API's /predict/ endpoint
            try:
                response = requests.post(f"{FLASK_API_URL}/predict/", json=input_data)

                if response.status_code == 200:
                    result = response.json()
                    if result['prediction'] == 1:
                        st.error("This transaction is predicted as FRAUD.")
                    else:
                        st.success("This transaction is predicted as LEGITIMATE.")
                else:
                    st.error(f"Error: Received status code {response.status_code}")
                    st.write(f"Response content: {response.content.decode('utf-8')}")

            except requests.exceptions.RequestException as e:
                st.error(f"Request failed: {e}")

    # If the user selects "Upload JSON", provide a file upload option
    elif input_method == "Upload JSON":
        uploaded_file = st.file_uploader("Choose a JSON file", type="json")

        if uploaded_file is not None:
            try:
                # Read and parse the uploaded JSON file
                input_data = json.load(uploaded_file)

                # Display the content of the uploaded file for user verification
                st.write("Uploaded JSON data:")
                st.json(input_data)

                # Button to submit the JSON file for prediction
                if st.button("Submit JSON for Prediction"):
                    try:
                        # Send a POST request to Flask API's /predict/ endpoint with the JSON data
                        response = requests.post(f"{FLASK_API_URL}/predict/", json=input_data)

                        if response.status_code == 200:
                            result = response.json()
                            if 'predictions' in result:
                                # Handle batch prediction
                                st.write("Batch Predictions:")
                                for i, pred in enumerate(result['predictions']):
                                    st.write(f"Transaction {i+1}: {'FRAUD' if pred == 1 else 'LEGITIMATE'}")
                            else:
                                # Handle single prediction
                                if result['prediction'] == 1:
                                    st.error("This transaction is predicted as FRAUD.")
                                else:
                                    st.success("This transaction is predicted as LEGITIMATE.")
                        else:
                            st.error(f"Error: Received status code {response.status_code}")
                            st.write(f"Response content: {response.content.decode('utf-8')}")
                    except requests.exceptions.RequestException as e:
                        st.error(f"Request failed: {e}")
            except json.JSONDecodeError:
                st.error("Invalid JSON file. Please check the format and try again.")

    # Button to fetch and display the prediction history
    if st.button("View Prediction History"):
        try:
            # Send a GET request to the Flask API's /history/ endpoint
            response = requests.get(f"{FLASK_API_URL}/get_history/")
            if response.status_code == 200:
                history_data = response.json().get("history", [])
                if history_data:
                    st.write("Prediction History:")
                    for entry in history_data:
                        st.json(entry)
                else:
                    st.write("No prediction history available.")
            else:
                st.error(f"Error: Received status code {response.status_code}")
                st.write(f"Response content: {response.content.decode('utf-8')}")
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")

with tabs[1]:
    st.header("Data Processing")

    st.write("This section is for administrators to select raw dataset files and process them for training.")

    # Get and display the current working directory
    current_directory = os.getcwd()
    st.write(f"**Load files from:** `{current_directory}/data_sources`")

    # List files in data_sources specific directory 
    data_dir = os.path.join(current_directory, "data_sources")  
    if os.path.exists(data_dir):
        available_files = os.listdir(data_dir)
    else:
        available_files = []

    if not available_files:
        st.error(f"No files found in the directory: {data_dir}")
    else:
        # Dropdown to select files for customer, fraud, and transactions datasets
        customer_file_path = st.selectbox("Select Customer Dataset", available_files)
        transactions_file_path = st.selectbox("Select Transactions Dataset", available_files)
        fraud_file_path = st.selectbox("Select Fraud Dataset", available_files)

        # Button to trigger the data processing
        if st.button("Process Dataset"):
            try:
                # Construct the full file paths
                customer_file_full_path = os.path.join(data_dir, customer_file_path)
                transactions_file_full_path = os.path.join(data_dir, transactions_file_path)
                fraud_file_full_path = os.path.join(data_dir, fraud_file_path)

                # Initialize the Raw_Data_Handler class
                raw_data_handler = Raw_Data_Handler()

                # Extract the datasets using the provided file paths
                customer_data, transactions_data, fraud_data = raw_data_handler.extract(
                    customer_file_full_path,
                    transactions_file_full_path,
                    fraud_file_full_path
                )

                # Validate that the datasets are not None
                if customer_data is None:
                    st.error(f"Error: Could not load customer dataset from {customer_file_full_path}")
                if transactions_data is None:
                    st.error(f"Error: Could not load transactions dataset from {transactions_file_full_path}")
                if fraud_data is None:
                    st.error(f"Error: Could not load fraud dataset from {fraud_file_full_path}")
                
                # If all datasets are successfully loaded, proceed to transformation
                if customer_data is not None and transactions_data is not None and fraud_data is not None:
                    # Transform the datasets using the transform method
                    processed_data = raw_data_handler.transform(
                        customer_information=customer_data,
                        transaction_information=transactions_data,
                        fraud_information=fraud_data
                    )

                    # Store the processed data in session state to persist it across interactions
                    st.session_state['processed_data'] = processed_data

                    # Show success message and display the first few rows of the processed dataset
                    st.success("Datasets processed successfully!")
                    
                    # Display sample data from processed datasets
                    st.write("Processed Data Sample:")
                    st.dataframe(processed_data.head())

                    # Get the earliest and latest transaction dates
                    min_date = processed_data['trans_date_trans_time'].min()
                    max_date = processed_data['trans_date_trans_time'].max()

                    # Store min and max dates in session state
                    st.session_state['min_date'] = min_date
                    st.session_state['max_date'] = max_date

            except Exception as e:
                st.error(f"Error processing datasets: {str(e)}")
        
        # Check if processed data and date range exist in session state
        if 'processed_data' in st.session_state and 'min_date' in st.session_state and 'max_date' in st.session_state:
            st.write(f"**Earliest Available Date:** {st.session_state['min_date']}")
            st.write(f"**Latest Available Date:** {st.session_state['max_date']}")

            # Input for start date, end date, and test set size
            start_date = st.date_input("Start Date", value=st.session_state['min_date'], min_value=st.session_state['min_date'], max_value=st.session_state['max_date'])
            end_date = st.date_input("End Date", value=st.session_state['max_date'], min_value=st.session_state['min_date'], max_value=st.session_state['max_date'])
            test_set_size = st.slider("Test Set Size (fraction)", min_value=0.1, max_value=0.5, value=0.2)

            # Button to create a new training dataset
            if st.button("Create Training Dataset"):
                try:
                    # Initialize the Dataset_Designer class
                    dataset_designer = Dataset_Designer()

                    # Use the processed data stored in session state
                    processed_raw_dataset = st.session_state['processed_data']

                    # Use the start and end date to sample the dataset
                    train_set, test_set = dataset_designer.sample(
                        raw_dataset=processed_raw_dataset,
                        test_size=test_set_size,
                        start_date=start_date,
                        end_date=end_date
                    )

                    # Show success message and display samples of the train and test sets
                    st.success("Training dataset created successfully!")
                    
                    st.write("Training Set Sample:")
                    st.dataframe(train_set.head())

                    st.write("Test Set Sample:")
                    st.dataframe(test_set.head())

                    # Store the save status of the train and test sets in session state
                    if 'save_train_test' not in st.session_state:
                        st.session_state['save_train_test'] = False

                    # Checkbox to save the train and test sets
                    st.session_state['save_train_test'] = st.checkbox(
                        "Save Training and Test Sets", value=st.session_state['save_train_test']
                    )

                    # Save the train and test sets if the checkbox is checked
                    if st.session_state['save_train_test']:
                        dataset_designer.load(output_filename="train_test_data")
                        st.write(f"Training and test sets saved at: {dataset_designer.storage_path}")

                except Exception as e:
                    st.error(f"Error creating training dataset: {str(e)}")
