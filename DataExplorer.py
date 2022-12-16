import os

from scipy.io import wavfile  # get the api


class DataExplorer:
    def __init__(self, data_path):

        self._data_path = os.path.join(data_path)
        # parametere used to extract sound features
        self._n_mels = 64
        self._frames = 5
        self._n_fft = 2048
        self._hop_length = 512
        self._power = 2.0

    def attempts(self):
        normal_signal_file = os.path.join(
            self._data_path, 'fan', 'id_00', 'normal', '00000100.wav')
        abnormal_signal_file = os.path.join(
            self._data_path, 'fan', 'id_00', 'abnormal', '00000100.wav')

        normal_signal, sr = wavfile.read(normal_signal_file)
        abnormal_signal, sr = wavfile.read(abnormal_signal_file)
        result = sr[0]
        print(
            f'The signals have a {normal_signal.shape} shape. At {sr} Hz, these are {normal_signal.shape[0]/sr:.0f}s signals')
