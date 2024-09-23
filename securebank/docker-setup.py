import os
import subprocess
import sys

def run_docker_compose():
    # Get the current working directory
    current_dir = os.getcwd()
    
    # Define the docker-compose command with the volume mount for the current directory
    command = [
        "docker-compose", 
        "up", 
        "--build"
    ]
    
    # Set the environment variable for the dataset path
    env = os.environ.copy()
    env['DATASET_PATH'] = current_dir
    
    print(f"Mounting current working directory {current_dir} to the container.")

    try:
        # Run the Docker Compose command
        subprocess.run(command, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_docker_compose()
