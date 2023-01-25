from preprocess_data import *

if __name__ == "__main__":

    # Call processing for rock db
    process_melody_chords(task='classification',
                          chords_path=r'../dataset/rock/pickle/rock_chords',
                          chords_csv=r'../dataset/rock/csv/rock_chords.csv',
                          dataset_addr=r'../dataset/rock/midi',
                          ints_path=r'../dataset/rock/pickle/ints_rock_dataset',
                          csv_labelled=r'../dataset/rock/csv/rock_labelled.csv',
                          final_csv_path=r'../dataset/rock/csv/rock_final.csv')

    # Call processing melody chords for classic db
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
                          dataset_addr=r'D:/midi_dataset/nes/full_db',
                          ints_path=r'../dataset/nes/pickle/ints_nes2',
                          csv_labelled=r'../dataset/nes/csv/nes_labelled2.csv',
                          final_csv_path=r'../dataset/nes/csv/nes_final_csv2.csv')

    # Preprocessing for classification between dataset
    # another logic from previous part
    # the final goal is create a pickle file with all ints representation of db songs with relative db label
    # having same number of files for each db
    # read the csv already created with the ints representation and use the csv as label
    read_db_csv_and_create_pickle(r'../dataset/nes/csv/nes_labelled2.csv', r'../dataset/rock/csv/rock_labelled.csv',
                                  r'../dataset/classic/csv/classic_labelled.csv')

    print("Processing is finished")

    # Call functions for extract statistics on midi length
    # Change for the dataset you want to extract stats on
    # medium_file_lenght(r'../dataset/rock/midi/')
    # medium_file_lenght_directories(r'../dataset/classic/midi/')
