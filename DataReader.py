import zipfile
import urllib.request
import os
import tools.utils


class DataDownloader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self._downloaded_zip = 'fan.zip'
        self._url_resources = 'https://zenodo.org/record/3384388/files/6_dB_fan.zip?download'

    @property
    def data_Dir(self):
        return self.data_dir

    def get_Data(self):
        if not os.path.exists(self.data_dir):
            print('Data directory does not exist, creating them.')
            os.makedirs(self.data_dir, exist_ok=True)

        # Checks if the dataset is already downloded and unzipped:
        first_file = os.path.join(self.data_dir, 'fan', 'id_00',
                                  'normal', '00000000.wav')
        if os.path.exists(first_file):
            print('=== Sound files found, no need to download them again. ===')

        else:
            print(
                '=== Downloading and unzipping the FAN file from the MIMII dataset website (~10 GB) ===')

            url = 'https://zenodo.org/record/3384388/files/6_dB_fan.zip?download'
            urllib.request.urlretrieve(url, self.downloaded_zip)

            # Checking file integrity: computing MD5 hash
            original_md5 = '0890f7d3c2fd8448634e69ff1d66dd47'
            downloaded_md5 = tools.utils.md5('/tmp/fan.zip')
            if original_md5 == downloaded_md5:
                with zipfile.ZipFile('output_document', 'r') as zip_ref:
                    zip_ref.extractall(self.data_dir)
            else:
                raise Exception(
                    'Downloaded file was corrupted, retry the download.')
