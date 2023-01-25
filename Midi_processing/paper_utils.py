from mido import MidiFile
import os
import numpy as np


def Average(lst):
    return sum(lst) / len(lst)


def medium_file_lenght(dataset_addr):
    mid = MidiFile('../dataset/nes/midi/nes_full_db/002_1943_TheBattleofMidway_03_04AirBattleA.mid')
    print(mid.length)
    midi_length = list()
    filez = list()

    for (dirpath, dirnames, filenames) in os.walk(dataset_addr):
        for file in filenames:
            mid = MidiFile(dataset_addr + '/' + file)
            midi_length.append(mid.length)

    # print(midi_lenght)
    medium_length = Average(midi_length)
    total_length = sum(midi_length)
    print("Medium Length of files in " + dataset_addr + ": ")
    print(medium_length)

    print("Toatl Length of files in " + dataset_addr + ": ")
    print(total_length)

    st_deviation = np.std(midi_length)
    print("Standard deviation of songs length " + dataset_addr + ": ")
    print(st_deviation)

    minimum_length = min(midi_length)
    print("Minimum song length " + dataset_addr + ": ")
    print(minimum_length)

    maximum_length = max(midi_length)
    print("Maximum song length " + dataset_addr + ": ")
    print(maximum_length)

    return medium_length


def medium_file_lenght_directories(dataset_addr):
    mid = MidiFile('../dataset/nes/midi/nes_full_db/002_1943_TheBattleofMidway_03_04AirBattleA.mid')
    print(mid.length)
    midi_length = list()
    filez = list()

    for (dirpath, dirnames, filenames) in os.walk(dataset_addr):
        for d in dirnames:
            folder = os.path.join(dirpath, d)
            for file in os.listdir(folder):
                try:
                    mid = MidiFile(folder + '/' + file)
                    midi_length.append(mid.length)
                except:
                    print("Bad midi: ", file)
                    continue

        # print(midi_lenght)
        medium_length = Average(midi_length)
        total_length = sum(midi_length)
        print("Medium Length of files in " + dataset_addr + ": ")
        print(medium_length)

        print("Toatl Length of files in " + dataset_addr + ": ")
        print(total_length)

        st_deviation = np.std(midi_length)
        print("Standard deviation of songs length " + dataset_addr + ": ")
        print(st_deviation)

        minimum_length = min(midi_length)
        print("Minimum song length " + dataset_addr + ": ")
        print(minimum_length)

        maximum_length = max(midi_length)
        print("Maximum song length " + dataset_addr + ": ")
        print(maximum_length)

        return medium_length
