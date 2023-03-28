from preprocess_data import *
import zipfile
import os

if __name__ == "__main__":

    # Extract and prepare datasets folder

    path_to_classic_zip_file = '../dataset/classic_db.zip'
    path_to_nes_zip_file = '../dataset/nes_db.zip'
    path_to_rock_zip_file = '../dataset/rock_db.zip'

    dir_to_classic_db = '../dataset/classic/'
    dir_to_nes_db = '../dataset/nes/'
    dir_to_rock_db = '../dataset/rock/'

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
                          chords_path=r'../dataset/rock/pickle/rock_chords',
                          chords_csv=r'../dataset/rock/csv/rock_chords.csv',
                          dataset_addr=r'../dataset/rock/midi',
                          ints_path=r'../dataset/rock/pickle/ints_rock_dataset',
                          csv_labelled=r'../dataset/rock/csv/rock_labelled.csv',
                          final_csv_path=r'../dataset/rock/csv/rock_final.csv')
    
    # # Call processing melody chords for classic db
    process_melody_chords(task='classification',
                          chords_path=r'../dataset/classic/pickle/classic_chords',
                          chords_csv=r'../dataset/classic/csv/classic_melody_chords.csv',
                          dataset_addr=r'../dataset/classic/midi',
                          ints_path=r'../dataset/classic/pickle/ints_classic_dataset',
                          csv_labelled=r'../dataset/classic/csv/classic_labelled.csv',
                          final_csv_path=r'../dataset/classic/csv/classic_final.csv')

    # Call processing for generation nesdb
    process_melody_chords(task='classification',
                          chords_path=r'../dataset/nes/pickle/nes_chords2',
                          chords_csv=r'../dataset/nes/csv/nes_chords2.csv',
                          dataset_addr=r'../dataset/nes/nes_db',
                          ints_path=r'../dataset/nes/pickle/ints_nes2',
                          csv_labelled=r'../dataset/nes/csv/nes_labelled2.csv',
                          final_csv_path=r'../dataset/nes/csv/nes_final_csv2.csv')

    # Preprocessing for classification between dataset
    # another logic from previous part
    # the final goal is create a pickle file with all ints representation of db songs with relative db label
    # having same number of files for each db
    # read the csv already created with the ints representation and use the csv as label
    read_db_csv_and_create_pickle(r'../dataset/nes/csv/nes_chords2.csv', r'../dataset/rock/csv/rock_labelled.csv',
                                  r'../dataset/classic/csv/classic_labelled.csv')
    
    print("Processing is finished")
