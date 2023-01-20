import os
import sys
import librosa
import librosa.display
import numpy as np
from PIL import Image
from tqdm import tqdm

def load_sound_file(wav_name, mono=False, channel=0):

    multi_channel_data, sampling_rate = librosa.load(wav_name, sr=None, mono=mono)
    signal = np.array(multi_channel_data)[channel, :]
    
    return signal, sampling_rate

def get_magnitude_scale(file, n_fft=1024, hop_length=512):

    signal, sampling_rate = load_sound_file(file)
    stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    dB = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    return dB

def extract_signal_features(signal, sr, n_mels=64, frames=5, n_fft=1024, hop_length=512):
    """
        signal (array of floats) - wyjście load_sound_file()
        sr (integer) częstość próbkowania
        n_mels (integer) liczba klatek MELa
        frames (integer) liczba okien na które zostanie pocięty spektrogram
        n_fft (integer)  długość sygnału do obliczenia krótkiej transformaty Fouriera
        hop_length (integer) inkrement okna
    """
    
    #Generowanie spektrogramu MEla
    mel_spectrogram = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    
    # Konwersja na decybele
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    # Generate an array of vectors as features for the current signal:
    features_vector_size = log_mel_spectrogram.shape[1] - frames + 1
    
    # Krótkie sygnały są pomijane:
    dims = frames * n_mels
    if features_vector_size < 1:
        return np.empty((0, dims), np.float32)
    
    # Budowa N okien danych(=frames) i budowa wektora cech:
    features = np.zeros((features_vector_size, dims), np.float32)
    for t in range(frames):
        features[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t:t + features_vector_size].T
        
    return features

def generate_dataset(files_list, n_mels=64, frames=5, n_fft=1024, hop_length=512):
    dims = n_mels * frames
    
    for index in tqdm(range(len(files_list)), desc='Ekstrakcja cech'):
        signal, sr = load_sound_file(files_list[index])
        
        features = extract_signal_features(
            signal, 
            sr, 
            n_mels=n_mels, 
            frames=frames, 
            n_fft=n_fft, 
            hop_length=hop_length
        )
        
        if index == 0:
            dataset = np.zeros((features.shape[0] * len(files_list), dims), np.float32)
            
        dataset[features.shape[0] * index : features.shape[0] * (index + 1), :] = features

    return dataset

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

def generate_spectrograms(list_files, output_dir, n_mels=64, n_fft=1024, hop_length=512):
    files = []
    
    for index in tqdm(range(len(list_files)), desc='kkkk'):
        file = list_files[index]
        path_components = file.split('/')
        
        machine_id, sound_type = path_components[-3], path_components[-2]
        wav_file = path_components[-1].split('.')[0]
        filename = sound_type + '-' + machine_id + '-' + wav_file + '.png'
        
        # Na przykład: train/normal/normal-id_02-00000259.png:
        filename = os.path.join(output_dir, sound_type, filename)

        if not os.path.exists(filename):
            # Tworzenie spektrogramu Mela
            signal, sr = load_sound_file(file)
            mels = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
            mels = librosa.power_to_db(mels, ref=np.max)

            # Zastosowanie skali minmax
            img = scale_minmax(mels, 0, 255).astype(np.uint8)
            img = np.flip(img, axis=0)
            img = 255 - img
            img = Image.fromarray(img)

            img.save(filename)

        files.append(filename)
        
    return files
