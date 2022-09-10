#!/usr/bin/env python3
# Copyright 2022, Educational Testing Service

from spacy.cli.download import download
import os

if __name__ == '__main__':
    print('Downloading Spacy and Coreferee Lexicons')
    download('en_core_web_sm')
    download('en_core_web_lg')
    download('en_core_web_trf')
    os.system("python3 -m coreferee install en")

