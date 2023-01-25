import numpy as np

import pandas as pd
import json
import os.path

import shutil

base_dir = os.getcwd()

with open(os.path.join(base_dir, "../Classification/lists/rpg.json"), "r") as fp:
    rpg = json.load(fp)
with open(os.path.join(base_dir, "../Classification/lists/fighting.json"), "r") as fp:
    fighting = json.load(fp)
with open(os.path.join(base_dir, "../Classification/lists/puzzle.json"), "r") as fp:
    puzzle = json.load(fp)
with open(os.path.join(base_dir, "../Classification/lists/sport.json"), "r") as fp:
    sport = json.load(fp)
with open(os.path.join(base_dir, "../Classification/lists/shooting.json"), "r") as fp:
    shooting = json.load(fp)

label_lists = puzzle + fighting + puzzle + sport + shooting

rock_artists = ['Eric Clapton', 'Queen', 'The Beatles', 'The Rolling Stones']

classic_artist = ['albanez', 'beethoven', 'mozart']

n_labels = 5


def label_nes_songs(dataset_addr, chords_csv, csv_labelled):
    # This directory has only the tracks already filtered by the muse preprocessing
    # that need to be labelled
    # in the end of the process the labelled tracks will be moved to other directory
    # and here will be left the unlabelled tracks

    labels = []
    # titles_labelled = list()
    # Process of labelling and moving files
    for (dirpath, dirnames, filenames) in os.walk(dataset_addr):
        # Assign labels to file
        for file in filenames:
            l = assign_label(file, n_labels)
            # Some file will be left without label
            # and here i move only the file with at least one label
            # from original folder to nes_labelled folder
            if not all(v == 0 for v in l):
                # os.replace(os.path.join(dirpath, file),
                #            os.path.join(r'../dataset/nes/midi/nes_labelled', file))
                src_path = os.path.join(dirpath, file)
                dst_path = os.path.join(r'D:/midi_dataset/nes/nes_labelled', file)
                shutil.copy(src_path, dst_path)
                # labels.append((file, l))
            labels.append((file, l))

    # dataset_training = r'../dataset/nes/midi/nes_labelled'

    # Add labels to the csv already created
    df = pd.read_csv(chords_csv, sep=';')

    # Need to sort the labels tuple to be in order with the csv
    d2 = {v: i for i, v in enumerate(list(df['filename']))}  # map elements to indexes

    labels = sorted(labels, key=lambda x: d2[x[0]])

    labels_values = [x[1] for x in labels]
    filename = [x[0] for x in labels]

    # Count as now do not work because of labels being a tuple
    count_labels(labels_values)

    df['label'] = labels_values
    # df['filename'] = filename

    df.to_csv(csv_labelled, sep=';', index=False)


# Check if a file is one of those of the selected games
def is_present(file_name, search_list=label_lists):
    present = False
    for name in search_list:
        if name in file_name:
            present = True
    return present


def assign_label(file_name, n_labels):
    labels = [0] * n_labels

    if is_present(file_name, rpg):
        labels[0] = 1
    if is_present(file_name, sport):
        labels[1] = 1
    if is_present(file_name, fighting):
        labels[2] = 1
    if is_present(file_name, shooting):
        labels[3] = 1
    if is_present(file_name, puzzle):
        labels[4] = 1

    return labels


def count_labels(labels):
    # Array containing the count for how many songs with each label are present
    # in this order : rpg,sport,fighting,shooting,puzzle
    labels_count = [0] * n_labels

    # labels = np.array(labels)

    # True if list is not nested so is in categorical form and we need to retransform in one hot
    if not any(isinstance(i, list) for i in labels):
        labels_one_hot = np.zeros((labels.size, labels.max() + 1))
        labels_one_hot[np.arange(labels.size), labels] = 1
    else:
        labels_one_hot = labels

    for x in labels_one_hot:
        for v in range(len(x)):
            if x[v] == 1:
                labels_count[v] = labels_count[v] + 1

    print("Labels count in this order, rpg,sport,fighting,shooting,puzzle : ", labels_count)

    return


# From the database in csv format return array with one hot encoded label ordered as in the csv file
def load_nes_label(csv_labelled):
    # prendo le label dal csv dove le ho annotate
    # train_data_csv = pd.read_csv(
    #    r"C:\Users\andri\Desktop\Tesi\Midi_Classification_and_Generation\nes_labelled.csv", sep=";")

    # train_data_csv = pd.read_csv(r"../Classification/nes_labelled.csv",
    #                             sep=";", converters={'label': eval})

    # train_data_y = train_data_csv['label'].tolist()

    df = pd.read_csv(csv_labelled, sep=';',
                     converters={'label': eval, 'int_tokens': eval})

    for index, row in df.iterrows():
        if not any(v == 1 for v in row['label']):
            df.drop(index, inplace=True)

    train_data_y = df['label'].tolist()
    train_data_x = df['int_tokens'].tolist()

    count_labels(train_data_y)

    # df.to_csv(r'../dataset/nes/csv/nes_labelled_songs.csv', sep=';', index=False)

    # Remove label where all elements are 0
    # they referred to unlabelled songs
    # for x in train_data_y:
    #     if not any(v == 1 for v in x):
    #         train_data_y.remove(x)

    return train_data_x, train_data_y


def label_from_directory(csv_path, dataset_addr, csv_label):
    df = pd.read_csv(csv_path, sep=';', converters={'int_tokens': eval})
    filez = df['filename'].values.tolist()
    labels = list()

    for (dirpath, dirnames, filenames) in os.walk(dataset_addr):
        # filez += [os.path.join(dirpath, file) for file in filenames]
        for d in dirnames:
            folder = os.path.join(dirpath, d)
            for file in os.listdir(folder):
                if file in filez:
                    artist = d
                    if 'rock' in dataset_addr:
                        l = assign_rock_label(artist)
                    if 'classic' in dataset_addr:
                        l = assign_classic_label(artist)
                    labels.append((file, l))

    # Need to sort the labels tuple to be in order with the csv
    d2 = {v: i for i, v in enumerate(list(df['filename']))}  # map elements to indexes

    labels = sorted(labels, key=lambda x: d2[x[0]])

    labels = [x[1] for x in labels]

    # Count as now do not work because of labels being a tuple
    count_labels(labels)

    df['label'] = labels

    df.to_csv(csv_label, sep=';', index=False)

    train_data_y = df['label'].tolist()
    train_data_x = df['int_tokens'].tolist()

    return train_data_x, train_data_y


def assign_rock_label(artist):
    labels = [0] * len(rock_artists)

    for idx, a in enumerate(rock_artists):
        if a == artist:
            labels[idx] = 1

    return labels


def assign_classic_label(artist):
    labels = [0] * len(classic_artist)

    for idx, a in enumerate(classic_artist):
        if a == artist:
            labels[idx] = 1

    return labels
