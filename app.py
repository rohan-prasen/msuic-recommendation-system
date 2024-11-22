import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Set up Spotify API credentials
client_id = '85820f40267d4a348fc36b42ec389dd9'
client_secret = '47b3cd77149b4f8bbabf605db65157d3'
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=client_id,
                                                           client_secret=client_secret))

# Load the dataset from the pkl file
data = joblib.load('music_recommendation_system.pkl')

# Define the necessary columns for processing
number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
               'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']

# Functions for song retrieval and recommendation
def find_song(name, year):
    song_data = defaultdict()
    results = sp.search(q='track:{} year:{}'.format(name, year), limit=1)
    if results['tracks']['items'] == []:
        return None
    results = results['tracks']['items'][0]
    track_id = results['id']
    audio_features = sp.audio_features(track_id)[0]
    song_data['name'] = [name]
    song_data['year'] = [year]
    song_data['explicit'] = [int(results['explicit'])]
    song_data['duration_ms'] = [results['duration_ms']]
    song_data['popularity'] = [results['popularity']]
    for key, value in audio_features.items():
        song_data[key] = value
    return pd.DataFrame(song_data)

def get_song_data(song, spotify_data):
    try:
        song_data = spotify_data[(spotify_data['name'] == song['name']) & (spotify_data['year'] == song['year'])].iloc[0]
        return song_data
    except IndexError:
        return find_song(song['name'], song['year'])

def get_mean_vector(song_list, spotify_data):
    song_vectors = []
    for song in song_list:
        song_data = get_song_data(song, spotify_data)
        if song_data is None:
            print(f"Warning: {song['name']} does not exist in Spotify or in the database.")
            continue
        song_vector = song_data[number_cols].values
        song_vectors.append(song_vector)
    song_matrix = np.array(list(song_vectors))
    return np.mean(song_matrix, axis=0)

def recommend_songs(song_list, spotify_data, n_songs=10):
    metadata_cols = ['name', 'year', 'artists']
    song_center = get_mean_vector(song_list, spotify_data)
    
    scaler = StandardScaler().fit(spotify_data[number_cols])
    scaled_data = scaler.transform(spotify_data[number_cols])
    scaled_song_center = scaler.transform(pd.DataFrame(song_center.reshape(1, -1), columns=number_cols))
    
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])
    
    # Get unique recommendations
    rec_songs = spotify_data.iloc[index].drop_duplicates(subset='name')
    
    # Filter out any songs that are already in the selected song list
    selected_song_names = {song['name'] for song in song_list}
    rec_songs = rec_songs[~rec_songs['name'].isin(selected_song_names)]
    
    return rec_songs[metadata_cols].to_dict(orient='records')  # Only return top 4 unique recommendations

# Streamlit UI
st.title("🎵 Music Recommendation System")
st.write("Search for a song and get recommendations based on it!")

# Search for a song on Spotify
song_name = st.text_input('Search for a song on Spotify', 'Blinding Lights')

if song_name:
    # Spotify Search
    results = sp.search(q=song_name, type='track', limit=4)
    
    if results['tracks']['items']:
        st.write(f"Search results for **{song_name}**:")

        # Create a grid layout for the tiles
        cols = st.columns(4)

        selected_song = None
        for i, track in enumerate(results['tracks']['items']):
            # Song metadata
            track_name = track['name']
            artist_name = track['artists'][0]['name']
            album_cover = track['album']['images'][0]['url']
            track_year = track['album']['release_date'][:4]

            with cols[i % 4]:  # Cycle through columns for each tile
                st.image(album_cover, width=100)  # Display album cover
                st.write(f"**Song:** {track_name}")
                st.write(f"**Artist:** {artist_name}")
                # Select button
                if st.button('Select', key=f"select_{i}"):
                    selected_song = {'name': track_name, 'year': int(track_year)}

        # If a song is selected, generate recommendations
        if selected_song:
            st.success(f"Selected song: {selected_song['name']} ({selected_song['year']})")
            
            # Get song recommendations
            recommendations = recommend_songs([selected_song], data)
            if recommendations:
                st.write(f"Top {len(recommendations)} recommendations based on **{selected_song['name']}**:")
                rec_df = pd.DataFrame(recommendations)
                rec_df = rec_df[['name', 'year', 'artists']]  # Ensure only required columns are displayed
                st.dataframe(rec_df)
            else:
                st.error("No recommendations found.")
    else:
        st.error("No songs found. Try searching for a different song.")
