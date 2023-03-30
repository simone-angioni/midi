from preprocess_data import *
import zipfile
import os

import sys

from pathlib import Path

path = Path(sys.path[0])
path_s = str(path.parent.absolute())

if __name__ == "__main__":

    # Extract and prepare datasets folder

    path_to_classic_zip_file = path_s + '/dataset/classic_db.zip'
    path_to_nes_zip_file = path_s + '/dataset/nes_db.zip'
    path_to_rock_zip_file = path_s + '/dataset/rock_db.zip'

    dir_to_classic_db = path_s + '/dataset/classic/'
    dir_to_nes_db = path_s + '/dataset/nes/'
    dir_to_rock_db = path_s + '/dataset/rock/'

    if not os.path.exists(dir_to_classic_db):
        with zipfile.ZipFile(path_to_classic_zip_file, 'r') as zip_ref:
            zip_ref.extractall(dir_to_classic_db)
            os.makedirs(dir_to_classic_db + 'pickle/')
            os.makedirs(dir_to_classic_db + 'csv/')

    if not os.path.exists(dir_to_rock_db):
        with zipfile.ZipFile(path_to_rock_zip_file, 'r') as zip_ref:
            zip_ref.extractall(dir_to_rock_db)
            os.makedirs(dir_to_rock_db + 'pickle/')
            os.makedirs(dir_to_rock_db + 'csv/')

    if not os.path.exists(dir_to_nes_db):
        with zipfile.ZipFile(path_to_nes_zip_file, 'r') as zip_ref:
            zip_ref.extractall(dir_to_nes_db)
            os.makedirs(dir_to_nes_db + 'pickle/')
            os.makedirs(dir_to_nes_db + 'csv/')

    # Call processing for rock db
    process_melody_chords(task='classification',
                          chords_path=path_s + r'../dataset/rock/pickle/rock_chords',
                          chords_csv=path_s + r'../dataset/rock/csv/rock_chords.csv',
                          dataset_addr=path_s + r'../dataset/rock/midi',
                          ints_path=path_s + r'../dataset/rock/pickle/ints_rock_dataset',
                          csv_labelled=path_s + r'../dataset/rock/csv/rock_labelled.csv',
                          final_csv_path=path_s + r'../dataset/rock/csv/rock_final.csv')
    
    # # # Call processing melody chords for classic db
    process_melody_chords(task='classification',
                          chords_path=path_s + r'../dataset/classic/pickle/classic_chords',
                          chords_csv=path_s + r'../dataset/classic/csv/classic_melody_chords.csv',
                          dataset_addr=path_s + r'../dataset/classic/midi',
                          ints_path=path_s + r'../dataset/classic/pickle/ints_classic_dataset',
                          csv_labelled=path_s + r'../dataset/classic/csv/classic_labelled.csv',
                          final_csv_path=path_s + r'../dataset/classic/csv/classic_final.csv')

    # Call processing for generation nesdb
    process_melody_chords(task='classification',
                          chords_path=path_s + r'../dataset/nes/pickle/nes_chords2',
                          chords_csv=path_s + r'../dataset/nes/csv/nes_chords2.csv',
                          dataset_addr=path_s + r'../dataset/nes/nes_db',
                          ints_path=path_s + r'../dataset/nes/pickle/ints_nes2',
                          csv_labelled=path_s + r'../dataset/nes/csv/nes_labelled2.csv',
                          final_csv_path=path_s + r'../dataset/nes/csv/nes_final_csv2.csv')

    # Preprocessing for classification between dataset another logic from previous part the final goal is create a
    # pickle file with all ints representation of db songs with relative db label having same number of files for
    # each db read the csv already created with the ints representation and use the csv as label
    read_db_csv_and_create_pickle(path_s + r'../dataset/nes/csv/nes_chords2.csv', path_s +
    r'../dataset/rock/csv/rock_labelled.csv', path_s + r'../dataset/classic/csv/classic_labelled.csv')

    print("Processing is finished")
