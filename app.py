import os
import subprocess
import sys

# This is a simple wrapper to launch the Streamlit app

if __name__ == "__main__":
    # Path to the actual app
    app_path = os.path.join("src", "app.py")
    
    # Check if the file exists
    if not os.path.exists(app_path):
        print(f"Error: Could not find {app_path}")
        sys.exit(1)
    
    # Launch the app using Streamlit
    print("Starting the Adaptive RAGL Chatbot...")
    subprocess.call(["streamlit", "run", app_path]) 