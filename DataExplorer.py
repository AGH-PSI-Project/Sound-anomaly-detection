import os
import tools.sound_tools
import matplotlib.pyplot as plt
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

        # parametere used to extract sound features
        self._n_mels = 64
        self._frames = 5
        self._n_fft = 2048
        self._hop_length = 512
        self._power = 2.0

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
        # init plot
        plt.style.use('Solarize_Light2')
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        blue, red = colors[1], colors[5]

        fig = plt.figure(figsize=(24, 6))
        plt.subplot(1, 3, 1)
        librosa.display.waveshow(self.normal_signal, sr=self.sr_n, alpha=0.5, color=blue,
                                 linewidth=0.5, label='Machine #id_00 - Normal signal')
        plt.title('Normal signal')

        plt.subplot(1, 3, 2)
        librosa.display.waveshow(self.abnormal_signal, sr=self.sr_ab, alpha=0.6, color=red,
                                 linewidth=0.5, label='Machine #id_00 - Abnormal signal')
        plt.title('Abnormal signal')

        plt.subplot(1, 3, 3)
        librosa.display.waveshow(self.abnormal_signal, sr=self.sr_ab, alpha=0.6,
                                 color=red, linewidth=0.5, label='Abnormal signal')
        librosa.display.waveshow(self.normal_signal, sr=self.sr_n, alpha=0.5,
                                 color=blue, linewidth=0.5, label='Normal signal')
        plt.title('Both signals')

        fig.suptitle(
            'Machine #id_00 - 2D representation of the wave forms', fontsize=16)

        plt.legend()
