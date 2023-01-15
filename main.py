from DataReader import DataDownloader
from DataExplorer import DataExplorer
import os


def main():
    data_dir_name = os.path.join('data', 'unprocessed')
    data_downloader = DataDownloader(data_dir_name)
    try:
        data_downloader.get_Data()
    except:
        print('Errors while trying to download sound records')
        exit(1)
    print('Resources are downloaded')

    data_explorer = DataExplorer(data_dir_name)
    data_explorer.load_sound_files()
    data_explorer.visualize_signals()


if __name__ == "__main__":
    main()
