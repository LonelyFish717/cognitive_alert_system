#!/bin/bash
echo "Installing dependencies..."
pip install -r requirements.txt
echo "Starting application..."
streamlit run app.py