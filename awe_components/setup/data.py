
from spacy.cli.download import download
import os

class data:

    def __init__(self):

        print('Downloading Spacy and Coreferee Lexicons')
        download('en_core_web_sm')
        download('en_core_web_lg')
        download('en_core_web_trf')
        os.system("python3 -m coreferee install en")


if __name__ == '__main__':
    data()
