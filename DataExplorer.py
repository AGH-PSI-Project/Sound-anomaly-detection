import os
import tools.sound_tools
import matplotlib.pyplot as plt
import numpy as np
import tools.sound_tools
# Sound management:
import librosa
import librosa.display
import IPython.display as ipd


class DataExplorer:
    def __init__(self, data_path):

        self._data_path = os.path.join(data_path)
        self.normal_signal = None
        self.sr_n = None
        self.abnormal_signal = None
        self.sr_ab = None

        # init plot
        plt.style.use('Solarize_Light2')
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        self.blue = colors[1]
        self.red = colors[5]

    def load_sound_files(self):
        normal_signal_file = os.path.join(
            self._data_path, 'fan', 'id_00', 'normal', '00000100.wav')
        abnormal_signal_file = os.path.join(
            self._data_path, 'fan', 'id_00', 'abnormal', '00000100.wav')

        self.normal_signal, self.sr_n = tools.sound_tools.load_sound_file(
            normal_signal_file)
        self.abnormal_signal, self.sr_ab = tools.sound_tools.load_sound_file(
            abnormal_signal_file)
        print(
            f'The signals have a {self.normal_signal.shape} shape. At {self.sr_n} Hz, these are {self.normal_signal.shape[0]/self.sr_n:.0f}s signals')
        print(
            f'The signals have a {self.abnormal_signal.shape} shape. At {self.sr_ab} Hz, these are {self.abnormal_signal.shape[0]/self.sr_ab:.0f}s signals')

    def visualize_signals(self):

        fig = plt.figure(figsize=(24, 6))
        plt.subplot(1, 3, 1)
        librosa.display.waveshow(self.normal_signal, sr=self.sr_n, alpha=0.5, color=self.blue,
                                 linewidth=0.5, label='Machine #id_00 - Normal signal')
        plt.title('Normal signal')

        plt.subplot(1, 3, 2)
        librosa.display.waveshow(self.abnormal_signal, sr=self.sr_ab, alpha=0.6, color=self.red,
                                 linewidth=0.5, label='Machine #id_00 - Abnormal signal')
        plt.title('Abnormal signal')

        plt.subplot(1, 3, 3)
        librosa.display.waveshow(self.abnormal_signal, sr=self.sr_ab, alpha=0.6,
                                 color=self.red, linewidth=0.5, label='Abnormal signal')
        librosa.display.waveshow(self.normal_signal, sr=self.sr_n, alpha=0.5,
                                 color=self.blue, linewidth=0.5, label='Normal signal')
        plt.title('Both signals')

        fig.suptitle(
            'Machine #id_00 - 2D representation of the wave forms', fontsize=16)

        plt.legend()

    def visualize_fourier_transofrm_partial(self):

        n_fft = 2048
        hop_length = 512

        D_normal = np.abs(librosa.stft(
            self.normal_signal[:n_fft], n_fft=n_fft, hop_length=n_fft + 1))

        D_abnormal = np.abs(librosa.stft(
            self.abnormal_signal[:n_fft], n_fft=n_fft, hop_length=n_fft + 1))

        fig = plt.figure(figsize=(12, 6))
        plt.plot(D_normal, color=self.blue, alpha=0.6,
                 label='Machine #id_00 - Normal signal')
        plt.plot(D_abnormal, color=self.red, alpha=0.6,
                 label='Machine #id_00 - Abnormal signal')
        plt.title('Fourier transform for the first 64ms window')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.xlim(0, 200)

    def visualize_spectograms_partial(self):

        n_fft = 2048
        hop_length = 512

        D_normal = np.abs(librosa.stft(
            self.normal_signal[:20*n_fft], n_fft=n_fft, hop_length=hop_length))
        normal_signal_file = os.path.join(
            self._data_path, 'fan', 'id_00', 'normal', '00000100.wav')
        dB_normal = tools.sound_tools.get_magnitude_scale(
            normal_signal_file)

        fig = plt.figure(figsize=(12, 6))
        librosa.display.specshow(D_normal, sr=self.sr_n, x_axis='time',
                                 y_axis='linear', cmap='viridis')
        plt.title(
            'Machine #id_00 - Normal signal\nShort Fourier Transform representation of the First 2560ms')
        plt.ylim(0, 500)
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.colorbar()
        plt.show()
