from DataReader import DataReader


def main():
    data_reader = DataReader()
    try:
        data_reader.get_Data()
    except:
        print('Errors ehile trying to download sound records')
        exit(1)
    print('Resources are downloaded')


if __name__ == "__main__":
    main()
