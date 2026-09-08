"""
Streamlit Community Cloud Entrypoint
Allows deploying seamlessly whether the main file path is set to 'app.py' or 'streamlit_app.py'.
"""
import os
import sys
import runpy

# Ensure root directory is on Python search path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Execute main app.py within the Streamlit runtime
target_app = os.path.join(current_dir, "app.py")
runpy.run_path(target_app, run_name="__main__")
