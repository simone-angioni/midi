from random import random

import pandas as pd
from tensorflow import keras

from Midi_Processing.labels_manager import load_nes_label

print('Loading needed modules. Please wait...')

import sys

sys.path.append(
    r'C:\Users\andri\Desktop\Tesi\Midi_Classification_and_Generation\libraries\tegridy-tools\tegridy-tools')

import TMIDIX

import numpy as np
from sklearn import preprocessing

from Midi_Processing.mini_muse_utils import *
from Midi_Processing.labels_manager import *


def process_melody_chords(task, chords_path, ints_path, dataset_addr, csv_labelled, final_csv_path, chords_csv):

    # MINI-MUSE PROCESSING

    # Create melody Chords representation and first csv with melody chords
    p_path = chords_path + '.pickle'

    if not os.path.exists(p_path):
        file_processed, melody_chords_f = create_melody_chords_dataset_2(dataset_addr, chords_path)

        df_file_processed = pd.DataFrame(columns=['filename', 'melody_chords'])
        df_file_processed['filename'] = file_processed
        df_file_processed['melody_chords'] = melody_chords_f

        df_file_processed.to_csv(chords_csv, index=False, sep=';')

    # Create ints representation from melody chords and add ints to csv with melody chords
    i_path = ints_path + '.pickle'

    if not os.path.exists(i_path):
        # Add int tokens to the csv from the melody chords
        melody_chords_f = TMIDIX.Tegridy_Any_Pickle_File_Reader(chords_path)
        # Inserting int in csv in inside this function
        ints_data = create_int_db_from_melody_chords(melody_chords_f, chords_csv, chords_csv)
        TMIDIX.Tegridy_Any_Pickle_File_Writer(ints_data, ints_path)

    if task == 'classification':

        # create the array for feed the classification network

        # Train x = array of max len  int token
        # paired with the train y for the labels

        # Label process for nes dataset
        if 'nes' in dataset_addr:
            print(dataset_addr)
            # Creating the directory with tracks labelled on the genre
            #path_nes_labelled = r'../dataset/nes/midi/nes_labelled'
            path_nes_labelled = r'D:/midi_dataset/nes/nes_labelled'
            dir = os.listdir(path_nes_labelled)
            # If the directory is empty the tracks aren't yet classified
            if len(dir) == 0:
                print('Labeling the nes database')
                #label_nes_songs(dataset_addr, chords_csv, csv_labelled)
                label_nes_songs(r'D:/midi_dataset/nes/full_db_pruned', chords_csv, csv_labelled)
            else:
                print("Directory with nes songs labelled already exists")

            # Add int tokens to the csv from the melody chords
            melody_chords_f = TMIDIX.Tegridy_Any_Pickle_File_Reader(chords_path)
            #csv = r'../dataset/nes/csv/nes_full_db.csv'
            #create_int_db_from_melody_chords(melody_chords_f, chords_csv, chords_csv)
            train_data_x, train_data_y = load_nes_label(csv_labelled)
            #train_list_x, train_list_y = build_list_of_max_n_tokens(melody_chords_f, train_data_y)
            pickle_int_path = r'../dataset/nes/pickle/nes_int_with_label2'
            TMIDIX.Tegridy_Any_Pickle_File_Writer((train_data_x, train_data_y), pickle_int_path)

        # Label process for rock, classic and intra dataset
        else:
            train_data_x, train_data_y = label_from_directory(csv_path=chords_csv,
                                                              dataset_addr=dataset_addr,
                                                              csv_label=csv_labelled)

            ints_path_labelled = ints_path + '_labelled'
            TMIDIX.Tegridy_Any_Pickle_File_Writer((train_data_x, train_data_y), ints_path_labelled)

        return

    # Training and generation preprocessing part
    else:
        # Check if the int tokens representation is already created
        if not os.path.exists(ints_path):
            # Read the melody chords format and transform it to ints representation
            melody_chords_f = TMIDIX.Tegridy_Any_Pickle_File_Reader(chords_path)
            train_list_x = create_int_db_from_melody_chords(melody_chords_f, chords_csv, final_csv_path)
            TMIDIX.Tegridy_Any_Pickle_File_Writer(train_list_x, ints_path)

        return train_list_x


# vector songs = song in int format
# can be divided in sub lists of max n tokens or a single list of int
def convert_vector_to_midi(vector_songs, ticks_per_quarter):
    # Manage when the vector song can contain more than one single vector
    # when in classification

    # if vector_song[0] is int:
    for idx, vector_song in enumerate(vector_songs):

        if len(vector_song) != 0:
            song = []
            song = vector_song
            song_f = []
            time = 0
            dur = 0
            vel = 0
            pitch = 0
            channel = 0

            for s in song:
                if s < 256:
                    time += s * 16

                else:
                    channel = s // 16 // 128

                    pitch = (s // 16) % 128

                    dur = ((s % 16) * 128) + 128

                    # Velocities for each channel:
                    if channel == 0:  # Piano
                        vel = 60
                    if channel == 1:  # Guitar
                        vel = 70
                    if channel == 2:  # Bass
                        vel = 60
                    if channel == 3:  # Violin
                        vel = 90
                    if channel == 4:  # Cello
                        vel = 100
                    if channel == 5:  # Harp
                        vel = 80
                    if channel == 6:  # Trumpet
                        vel = 100
                    if channel == 7:  # Clarinet
                        vel = 100
                    if channel == 8:  # Flute
                        vel = 100
                    if channel == 9:  # Drums
                        vel = 80
                    if channel == 10:  # Choir
                        vel = 110

                    song_f.append(['note', time, dur, channel, pitch, vel])

        # else:
        #    song_f = vector_song

        detailed_stats = TMIDIX.Tegridy_SONG_to_MIDI_Converter(song_f,
                                                               output_signature='Yoda',
                                                               output_file_name=r'..\converted_midi\converted_midi' + str(
                                                                   idx),
                                                               track_name='converted midi' + str(idx),
                                                               list_of_MIDI_patches=[0, 24, 32, 40, 42, 46, 56, 71, 73,
                                                                                     0,
                                                                                     53, 0, 0, 0, 0, 0],
                                                               number_of_ticks_per_quarter=ticks_per_quarter)

        for key, value in detailed_stats.items():
            print('=' * 70)
            print(key, '|', value)

    return song_f


def read_db_csv_and_create_pickle(nes_csv, rock_csv, classic_csv):

    nes_df = pd.read_csv(nes_csv, sep=';', converters={'int_tokens': eval})
    rock_df = pd.read_csv(rock_csv, sep=';', converters={'int_tokens': eval})
    classic_df = pd.read_csv(classic_csv, sep=';', converters={'int_tokens': eval})

    train_data_x_nes = nes_df['int_tokens'].tolist()
    train_data_x_rock = rock_df['int_tokens'].tolist()
    train_data_x_classic = classic_df['int_tokens'].tolist()

    nes_files = nes_df['filename'].values.tolist()
    rock_files = rock_df['filename'].values.tolist()
    classic_files = classic_df['filename'].values.tolist()

    # Reduce all the data to max 100 records
    train_data_x_nes = train_data_x_nes[:250]
    train_data_x_rock = train_data_x_rock[:50]
    train_data_x_classic = train_data_x_classic[:100]

    train_data_x = train_data_x_nes + train_data_x_rock + train_data_x_classic

    nes_files = nes_files[:250]
    rock_files = rock_files[:50]
    classic_files = classic_files[:100]

    # Create label for each song in this order: NES, rock , classic
    nes_labelz = create_label_for_filename(nes_files, [1, 0, 0])
    rock_labelz = create_label_for_filename(rock_files, [0, 1, 0])
    classic_labelz = create_label_for_filename(classic_files, [0, 0, 1])

    labels_list = nes_labelz + rock_labelz + classic_labelz

    datasets_df = pd.DataFrame(columns=['filename', 'int_tokens', 'label'])
    datasets_df['filename'] = nes_files + rock_files + classic_files
    datasets_df['int_tokens'] = train_data_x
    datasets_df['label'] = labels_list

    datasets_df.to_csv(r"../dataset/db_merged/db_merged.csv", sep=';', index=False)

    TMIDIX.Tegridy_Any_Pickle_File_Writer((train_data_x, labels_list), r'../dataset/db_merged/ints_db_merged_labelled')

    return


def create_label_for_filename(filez, label):
    labelz = []
    for f in filez:
        labelz.append(label)
    return labelz
