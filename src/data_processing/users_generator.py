import random
import pandas as pd
import os

def generate_user_data(num_users=100):
    """
    Generates random user data and saves it into a CSV file named 'users.csv' 
    in the specified path '../../datasets/'.

    Parameters:
    num_users (int): Number of users to generate. Default is 100.
    
    The generated data includes the following columns:
    - user_id
    - name
    - last_name
    - nationality
    - age
    - gender
    - mood
    - favorite_movies
    - recently_watched_movies
    """
    
    # Sample data to generate random users
    names = ['John', 'Alice', 'Bob', 'Emma', 'Liam', 'Sophia', 'William', 'Mia', 'James', 'Charlotte']
    last_names = ['Smith', 'Johnson', 'Williams', 'Jones', 'Brown', 'Davis', 'Miller', 'Wilson', 'Moore', 'Taylor']
    nationalities = ['Cuban', 'American', 'Canadian', 'Mexican', 'Spanish', 'Argentinian', 'Brazilian', 'French', 'German', 'Italian']
    moods = ['happy', 'sad', 'excited', 'romantic', 'melancholic']
    genders = ['Male', 'Female', 'Non-binary']
    movies = ['Inception', 'The Matrix', 'Titanic', 'The Godfather', 'Pulp Fiction', 'Forrest Gump', 'The Shawshank Redemption', 'Interstellar', 'Gladiator', 'The Dark Knight']

    # List to hold all user data
    user_data = []

    for user_id in range(1, num_users + 1):
        user = {
            'user_id': user_id,
            'name': random.choice(names),
            'last_name': random.choice(last_names),
            'nationality': random.choice(nationalities),
            'age': random.randint(18, 80),
            'gender': random.choice(genders),
            'mood': random.choice(moods),
            'favorite_movies': ', '.join(random.sample(movies, 3)),  # Select 3 random favorite movies
            'recently_watched_movies': ', '.join(random.sample(movies, 5))  # Select 5 random recently watched movies
        }
        user_data.append(user)
    
    # Convert the list of dictionaries to a DataFrame
    users_df = pd.DataFrame(user_data)
    
    # Define the path to save the CSV
    path = '../../datasets/'
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Save the DataFrame to a CSV file
    users_df.to_csv(os.path.join(path, 'users.csv'), index=False)
    print(f"User data has been successfully saved to '{os.path.join(path, 'users.csv')}'.")

# Generate the user data and save it to '../../datasets/users.csv'
generate_user_data()
