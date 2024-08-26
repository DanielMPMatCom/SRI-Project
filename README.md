# SRI-Project

# Hybrid Recommendation System for Cuban Audiovisual Products

## Overview

This project is a hybrid recommendation system designed to recommend audiovisual products, specifically tailored for the Cuban audience. The system combines collaborative filtering and content-based filtering techniques to improve the relevance of recommendations. Additionally, it adapts recommendations to the Cuban context by incorporating local preferences.

## Authors
- Daniel Machado Pérez C311
- Osvaldo R. Moreno Prieto C311
- Daniel Toledo Martínez C311

## Problem Statement

In an environment where users are presented with a vast number of audiovisual products, it is crucial to have a system that can efficiently recommend content that matches user preferences. This project aims to develop a hybrid recommendation system that leverages multiple recommendation techniques to deliver more accurate and contextually relevant recommendations.

## Requirements

- Python 3.8+
- See `requirements.txt` for a list of required Python packages.

## APIs

This project may use external APIs for data retrieval or additional functionalities, which should be specified in the respective modules.

## How to Run the Project

1. **Clone the repository:**
    ```bash
    git clone https://github.com/DanielMPMatCom/SRI-Project.git
    cd SRI-Project
    ```

2. **Set up the virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the project:**
    ```bash
    ./startup.sh
    ```

## Project Structure

. ├── data/ │ ├── users.csv │ ├── items.csv ├── datasets/ │ ├── users.csv ├── src/ │ ├── config.py │ ├── main.py │ ├── data_processing/ │ │ ├── data_loader.py │ │ ├── preprocessing.py │ ├── recommendation/ │ │ ├── collaborative_filtering.py │ │ ├── content_based.py │ │ ├── hybrid.py │ │ ├── cuban_context.py │ ├── evaluation/ │ │ ├── metrics.py ├── tests/ │ ├── test_collaborative_filtering.py │ ├── test_content_based.py │ ├── test_hybrid.py │ ├── test_cuban_context.py │ ├── test_metrics.py ├── startup.sh ├── requirements.txt ├── README.md


## New Features

### 1. User Data Generation

A script has been added to generate random user data and save it into a CSV file named `users.csv`. This file contains columns such as `name`, `last_name`, `nationality`, `age`, `gender`, `mood`, `favorite_movies`, and `recently_watched_movies`. The generated data is saved in the path `../../datasets/`.

#### How to Generate User Data

The user data generation script is part of the project's data preparation steps. The following steps outline how to generate the user data:

1. **Generate User Data:**
    - The script automatically generates random data for a specified number of users.
    - It saves the resulting data in a CSV file in the `../../datasets/` directory.

2. **User Data File Structure:**
    - The CSV file includes the following columns:
      - `name`: First name of the user.
      - `last_name`: Last name of the user.
      - `nationality`: Nationality of the user.
      - `age`: Age of the user.
      - `gender`: Gender of the user.
      - `mood`: Current mood of the user.
      - `favorite_movies`: A list of the user's favorite movies.
      - `recently_watched_movies`: A list of the last 5 movies watched by the user.

### 2. Enhanced Recommendation Features

- **Emotion-Based Recommendations:** The system now incorporates an analysis of user emotions, adjusting recommendations based on the user's current mood (e.g., comedies for happiness, action for excitement, dramas for sadness).
- **Avoiding Repeated Recommendations:** The system tracks movies already watched by the user and avoids recommending them again as primary options. This feature is designed to improve user experience by providing more relevant recommendations.

## License

This project is licensed under the MIT License.
