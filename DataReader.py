# Python libraries:
import os
import urllib.request
import zipfile
from Crypto.Hash import MD5

# Sound management:
import librosa
import librosa.display
import IPython.display as ipd


def get_MD5(file_path):

    chunk_size = 8192
    h = MD5.new()
    with open(file_path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if len(chunk):
                h.update(chunk)
            else:
                break
    return h.hexdigest()


def get_Data():

    data_dir = 'data'
    if not os.path.exists(data_dir):
        print('Data directory does not exist, creating them.')
        os.makedirs(data_dir, exist_ok=True)

    # Checks if the dataset is already downloded and unzipped:
    first_file = os.path.join(data_dir, 'fan', 'id_00',
                              'normal', '00000000.wav')
    if os.path.exists(first_file):
        print('=== Sound files found, no need to download them again. ===')

    else:
        print('=== Downloading and unzipping the FAN file from the MIMII dataset website (~10 GB) ===')
        output_document = 'fan.zip'
        url = 'https://zenodo.org/record/3384388/files/6_dB_fan.zip?download'
        #url = 'https://stackoverflow.com/questions/3451111/unzipping-files-in-python'
        urllib.request.urlretrieve(url, output_document)

        # Checking file integrity: computing MD5 hash
        original_md5 = '0890f7d3c2fd8448634e69ff1d66dd47'
        if original_md5 == get_MD5(output_document):
            with zipfile.ZipFile('output_document', 'r') as zip_ref:
                zip_ref.extractall(data_dir)
        else:
            raise Exception(
                'Downloaded file was corrupted, retry the download.')


get_Data()
