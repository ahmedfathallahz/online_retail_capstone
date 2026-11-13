#!/bin/bash
# Launch the Streamlit Capstone Dashboard
cd "$(dirname "$0")/streamlit_app"
streamlit run app.py --server.headless true
