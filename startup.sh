#!/bin/bash

# This script sets up and runs the recommendation system project.

# Activate virtual environment if exists
if [ -d "venv" ]; then
  source venv/bin/activate
fi

# Run the main application (assuming main.py is the entry point)
python src/main.py
