import pandas as pd
import os

def process_disney_data(disney_path, output_path):
    """
    Processes Disney data and appends or creates a CSV file with the selected information.
    
    Parameters:
    disney_path (str): Path to the Disney dataset.
    output_path (str): Path to the output CSV file.
    """
    # Read the Disney file
    try:
        disney_df = pd.read_csv(disney_path)
    except FileNotFoundError:
        print(f"The file {disney_path} was not found.")
        return
    except pd.errors.EmptyDataError:
        print(f"The file {disney_path} is empty.")
        return
    
    # Extract relevant data
    data = {
        'Title': disney_df['movie_title'] if 'movie_title' in disney_df.columns else pd.Series([''] * len(disney_df)),
        'Year': pd.to_datetime(disney_df['release_date'], errors='coerce').dt.year if 'release_date' in disney_df.columns else pd.Series([''] * len(disney_df)),
        'Genre': disney_df['genre'] if 'genre' in disney_df.columns else pd.Series([''] * len(disney_df)),
    }
    
    # Create DataFrame with the selected data
    processed_df = pd.DataFrame(data)
    
    # Check if the output file already exists
    if os.path.exists(output_path):
        print(f"The file {output_path} already exists. New data will be appended.")
        try:
            existing_df = pd.read_csv(output_path)
        except pd.errors.EmptyDataError:
            print(f"The file {output_path} is empty. A new file will be created.")
            existing_df = pd.DataFrame()
        combined_df = pd.concat([existing_df, processed_df], ignore_index=True)
    else:
        print(f"The file {output_path} does not exist. A new file will be created.")
        combined_df = processed_df
    
    # Save the combined DataFrame to the output file
    try:
        combined_df.to_csv(output_path, index=False)
        print(f"Data successfully saved to {output_path}")
    except Exception as e:
        print(f"Error saving data to {output_path}: {e}")

def process_imdb_data(imdb_path, output_path):
    """
    Processes IMDb data and appends or creates a CSV file with the selected information.
    
    Parameters:
    imdb_path (str): Path to the IMDb dataset.
    output_path (str): Path to the output CSV file.
    """
    # Read the IMDb file
    imdb_df = pd.read_csv(imdb_path)
    
    # Extract relevant data and create the 'Cast' column
    data = {
        'Title': imdb_df['Series_Title'] if 'Series_Title' in imdb_df.columns else pd.Series([''] * len(imdb_df)),
        'Year': pd.to_numeric(imdb_df['Released_Year'], errors='coerce') if 'Released_Year' in imdb_df.columns else pd.Series([''] * len(imdb_df)),
        'Genre': imdb_df['Genre'] if 'Genre' in imdb_df.columns else pd.Series([''] * len(imdb_df)),
        'Overview': imdb_df['Overview'] if 'Overview' in imdb_df.columns else pd.Series([''] * len(imdb_df)),
        'IMDB_Rating': imdb_df['IMDB_Rating'] if 'IMDB_Rating' in imdb_df.columns else pd.Series([''] * len(imdb_df)),
        'Director': imdb_df['Director'] if 'Director' in imdb_df.columns else pd.Series([''] * len(imdb_df)),
        'Cast': imdb_df[['Star1', 'Star2', 'Star3', 'Star4']].apply(lambda x: ', '.join(x.dropna().astype(str)), axis=1) if {'Star1', 'Star2', 'Star3', 'Star4'}.issubset(imdb_df.columns) else pd.Series([''] * len(imdb_df)),
    }
    
    # Create DataFrame with the selected data
    processed_df = pd.DataFrame(data)
    
    # Check if the output file already exists
    if os.path.exists(output_path):
        print(f"The file {output_path} already exists. New data will be appended.")
        existing_df = pd.read_csv(output_path)
        combined_df = pd.concat([existing_df, processed_df], ignore_index=True)
    else:
        print(f"The file {output_path} does not exist. A new file will be created.")
        combined_df = processed_df
    
    # Save the combined DataFrame to the output file
    combined_df.to_csv(output_path, index=False)
    print(f"Data successfully saved to {output_path}")

def process_netflix_data(netflix_path, output_path):
    """
    Processes Netflix data and appends or creates a CSV file with the selected information.
    
    Parameters:
    netflix_path (str): Path to the Netflix dataset.
    output_path (str): Path to the output CSV file.
    """
    # Read the Netflix file
    netflix_df = pd.read_csv(netflix_path)
    
    # Extract relevant data
    data = {
        'Title': netflix_df['title'] if 'title' in netflix_df.columns else pd.Series([''] * len(netflix_df)),
        'Year': netflix_df['release_year'] if 'release_year' in netflix_df.columns else pd.Series([''] * len(netflix_df)),
        'Genre': netflix_df['listed_in'] if 'listed_in' in netflix_df.columns else pd.Series([''] * len(netflix_df)),
        'Overview': netflix_df['description'] if 'description' in netflix_df.columns else pd.Series([''] * len(netflix_df)),
        'IMDB_Rating': pd.Series([''] * len(netflix_df)),  # IMDb Rating is not available, so leave it empty
        'Director': netflix_df['director'] if 'director' in netflix_df.columns else pd.Series([''] * len(netflix_df)),
        'Cast': netflix_df['cast'] if 'cast' in netflix_df.columns else pd.Series([''] * len(netflix_df)),
        'Type': netflix_df['type'] if 'type' in netflix_df.columns else pd.Series([''] * len(netflix_df)),
        'Country': netflix_df['country'] if 'country' in netflix_df.columns else pd.Series([''] * len(netflix_df)),
    }
    
    # Create DataFrame with the selected data
    processed_df = pd.DataFrame(data)
    
    # Check if the output file already exists
    if os.path.exists(output_path):
        print(f"The file {output_path} already exists. New data will be appended.")
        existing_df = pd.read_csv(output_path)
        combined_df = pd.concat([existing_df, processed_df], ignore_index=True)
    else:
        print(f"The file {output_path} does not exist. A new file will be created.")
        combined_df = processed_df
    
    # Save the combined DataFrame to the output file
    combined_df.to_csv(output_path, index=False)
    print(f"Data successfully saved to {output_path}")

# File paths
disney_path = "../../datasets/Disney Movies/disney_movies.csv"
imdb_path = "../../datasets/IMDB Movies Dataset/imdb_top_1000.csv"
netflix_path = "../../datasets/Netflix Movies and TV Shows/netflix_titles.csv"
output_path = "../../datasets/movies_series_dataset.csv"

# Process the datasets
process_disney_data(disney_path, output_path)
process_imdb_data(imdb_path, output_path)
process_netflix_data(netflix_path, output_path)

def clean_dataset(file_path):
    """
    Cleans the dataset by merging duplicate rows based on 'Title' and 'Year' columns.

    Parameters:
    file_path (str): Path to the CSV file to be cleaned.
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Create a list to store the combined rows
    combined_rows = []

    # Group by 'Title' and 'Year'
    grouped = df.groupby(['Title', 'Year'], as_index=False)

    for _, group in grouped:
        if len(group) > 1:
            # Combine the rows in the group
            combined_row = group.iloc[0].copy()
            for _, row in group.iterrows():
                for col in df.columns:
                    if pd.isna(combined_row[col]) or combined_row[col] == "":
                        combined_row[col] = row[col]
                    elif row[col] != combined_row[col] and pd.notna(row[col]):
                        combined_row[col] += f", {row[col]}"
            combined_rows.append(combined_row)
        else:
            # If no duplicate rows, simply add the row to the list
            combined_rows.append(group.iloc[0])

    # Convert the list of combined rows into a DataFrame
    combined_df = pd.DataFrame(combined_rows, columns=df.columns)

    # Save the cleaned DataFrame to the original file
    combined_df.to_csv(file_path, index=False)

# Path to the output file
file_path = "../../datasets/movies_series_dataset.csv"

# Call the function to clean the dataset
clean_dataset(file_path)
