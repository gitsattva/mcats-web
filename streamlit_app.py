import matplotlib.pyplot as plt
import streamlit as st
import librosa

"""
# Welcome to Streamlit!

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:

If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
"""

def normalize_volume(music_file):
    audio, sr = librosa.load(music_file, offset=30.0, duration=30.0)
    audio_norm = librosa.util.normalize(audio, axis=0)
    return audio_norm, sr

music_file = st.file_uploader("Choose a music file")

if music_file is not None:
    audio_norm, sr = normalize_volume(music_file)
    audio_stft = librosa.stft(audio_norm)
    audio_db = librosa.amplitude_to_db(abs(audio_stft))
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(audio_db, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()
    st.write("Here is the spectrogram!")
