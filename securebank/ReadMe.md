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
