# from random import random
from random import shuffle

import os

import numpy as np
import tqdm as tqdm
from tqdm import tqdm

import sys

import pandas as pd

# sys.path.append(
#     r'C:\Users\andri\Desktop\Tesi\Midi_Classification_and_Generation\libraries\tegridy-tools\tegridy-tools')

import TMIDIX


# Questo era una prova di un nuovo preprocessing abbandonata si potrebbe rivedere
def create_melody_chords_dataset_2(dataset_addr, pickle_path):
    sorted_or_random_file_loading_order = False  # Sorted order is NOT usually recommended
    dataset_ratio = 1  # Change this if you need more data

    print('TMIDIX MIDI Processor')
    print('Starting up...')
    ###########

    files_count = 0

    gfiles = []

    melody_chords_f = []

    print('Loading MIDI files...')
    print('This may take a while on a large dataset in particular.')

    filez = list()

    for (dirpath, dirnames, filenames) in os.walk(dataset_addr):
        filez += [os.path.join(dirpath, file) for file in filenames]

    print('=' * 70)

    if filez == []:
        print('Could not find any MIDI files. Please check Dataset dir...')
        print('=' * 70)

    if sorted_or_random_file_loading_order:
        print('Sorting files...')
        filez.sort()
        print('Done!')
        print('=' * 70)
    else:
        print('Randomizing file list...')
        shuffle(filez)

    stats = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    middles_stats = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    file_processed = list()

    n_instruments_list = list()

    print('Processing MIDI files. Please wait...')
    for f in tqdm(filez[:int(len(filez) * dataset_ratio)]):
        try:
            fn = os.path.basename(f)
            fn1 = fn.split('.')[0]

            # print('Loading MIDI file...')
            score = TMIDIX.midi2ms_score(open(f, 'rb').read())

            events_matrix = []

            itrack = 1

            patches = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

            instruments_found = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

            patch_map = [[0, 1, 2, 3, 4, 5, 6, 7],  # Piano
                         [24, 25, 26, 27, 28, 29, 30],  # Guitar
                         [32, 33, 34, 35, 36, 37, 38, 39],  # Bass
                         [40, 41],  # Violin
                         [42, 43],  # Cello
                         [46],  # Harp
                         [56, 57, 58, 59, 60],  # Trumpet
                         [71, 72],  # Clarinet
                         [73, 74, 75],  # Flute
                         [-1],  # Fake Drums
                         [52, 53],  # Choir
                         [16, 17, 18, 19, 20]  # Organ
                         ]

            while itrack < len(score):
                for event in score[itrack]:
                    if event[0] == 'note' or event[0] == 'patch_change':
                        events_matrix.append(event)
                itrack += 1

            events_matrix.sort(key=lambda x: x[1])

            events_matrix1 = []
            for event in events_matrix:
                # Patch change = program change
                if event[0] == 'patch_change':
                    # Patches contiene i canali, ovvero gli indici del vettore rappresentano i canali
                    # Qui si va a selezionare il canale dell'evento -> event[2] nel patch change
                    # e si va ad aggiornare con il valore del program number -> event[3] nel patch change
                    patches[event[2]] = event[3]

                if event[0] == 'note':
                    # qui event[3] indica il canale dove la nota dev'essere suonata
                    # stiamo aggiungendo all'evento nota il numero dello strumento
                    # estratto dal patch change o program change
                    event.extend([patches[event[3]]])
                    once = False

                    for p in patch_map:
                        # event[6] è il program number = strumento con cui si deve suonare la nota
                        # appena aggiunto al passo precedente
                        if event[6] in p and event[3] != 9:  # Except the drums
                            # Aggiorna il canale dove deve suonare la nota con l'indice dello strumento
                            # nel patch map
                            # Così facendo facciamo corrispondere ogni canale ad uno strumento in modo
                            # da risparmiare spazio
                            event[3] = patch_map.index(p)
                            once = True

                    if not once and event[3] != 9:  # Except the drums
                        event[3] = 0  # All other instruments/patches channel
                        event[5] = max(80, event[5])

                    if event[3] < 12:  # We won't write chans 11-16 for now...
                        events_matrix1.append(event)
                        stats[event[3]] += 1
                        instruments_found[event[3]] = 1

            # Sorting...
            events_matrix1.sort(key=lambda x: (x[1], x[3]))

            # recalculating timings
            for e in events_matrix1:
                e[1] = int(e[1] / 16)
                e[2] = int(e[2] / 32)

            # final processing...
            melody_chords = []

            # Take first note
            pe = events_matrix1[0]
            for e in events_matrix1:
                time = max(0, min(127, e[1] - pe[1]))
                dur = max(1, min(127, e[2]))
                cha = max(0, min(11, e[3]))
                ptc = max(1, min(127, e[4]))
                vel = max(19, min(127, e[5]))

                div_vel = int(vel / 19)

                chan_vel = (cha * 11) + div_vel

                #melody_chords_f.extend([chan_vel, time + 128, dur + 256, ptc + 384])
                melody_chords.append([chan_vel, time + 128, dur + 256, ptc + 384])

                middles_stats[cha] += 1
                pe = e

            # process only files with n notes or more
            if len(melody_chords) >= 128:

                melody_chords_f.append(melody_chords)

                files_count += 1
                file_processed.append(fn)

                n_instruments = instruments_found.count(1)
                n_instruments_list.append(n_instruments)

                # Move files processed in specific folder for nes processing
                if 'nes' in dataset_addr:
                    os.replace(f, os.path.join(r'D:/midi_dataset/nes/full_db_pruned', fn))
            else:
                print("skipped midi " + fn + "of length" + str(len(melody_chords)))
                #os.remove(f)
                continue

        except KeyboardInterrupt:
            print('Saving current progress and quitting...')
            break

        except:
            print('Bad MIDI:', f)
            os.remove(f)
            continue

    print('=' * 70)

    print('Done!')
    print('=' * 70)

    print('Resulting Stats:')
    print('=' * 70)
    print('Total MIDI Excerpts:', files_count)
    print('=' * 70)

    print('Piano:', middles_stats[0])
    print('Guitar:', middles_stats[1])
    print('Bass:', middles_stats[2])
    print('Violin:', middles_stats[3])
    print('Cello:', middles_stats[4])
    print('Harp:', middles_stats[5])
    print('Trumpet:', middles_stats[6])
    print('Clarinet:', middles_stats[7])
    print('Flute:', middles_stats[8])
    print('Drums:', middles_stats[9])
    print('Choir:', middles_stats[10])

    print('=' * 70)

    instruments_count = {}
    for x in range(1, 15):
        instruments_count["{0}_instrument_song".format(x)] = n_instruments_list.count(x)

    TMIDIX.Tegridy_Any_Pickle_File_Writer(melody_chords_f, pickle_path)

    return file_processed, melody_chords_f


def create_int_db_from_melody_chords(melody_chords_f, csv_melody_chords, final_csv_path):
    randomize_dataset = False

    print('=' * 70)
    print('Prepping INTs dataset...')

    if randomize_dataset:
        print('=' * 70)
        print('Randomizing the dataset...')
        shuffle(melody_chords_f)
        print('Done!')

    print('=' * 70)
    print('Processing the dataset...')

    ints_list = []

    for m in tqdm(melody_chords_f):
        # if len(m) != 256:
        #     print('Error')
        # else:
        ints_list.extend([0])
        for mm in m:
            ints_list.extend(mm)

    # Here the min length of ints list is 513 (512 token + 0 token in the beginning)
    data_for_csv = split_list_on_value(ints_list, 0)

    chunks = []

    # Split the ints to have max length 512
    for idx, d in enumerate(data_for_csv):
        chunks = split_int_array_in_512chunks(d)
        # chunks = [d[x:x + 512] for x in range(0, len(d), 512)]
        # for i, c in enumerate(chunks):
        #     # Pad chunks shorter than 512 with 0
        #     if len(c) < 512:
        #         c = np.pad(c, (0, 512 - len(c) % 512), 'constant')
        #         c = c.tolist()
        #         chunks[i] = c
        data_for_csv[idx] = chunks

    df = pd.read_csv(csv_melody_chords, sep=';')

    df['int_tokens'] = data_for_csv

    df.to_csv(final_csv_path, index=False, sep=';')

    print('Done!')
    print('=' * 70)

    print('Total INTs:', len(ints_list))
    print('Minimum INT:', min(ints_list))
    print('Maximum INT:', max(ints_list))
    print('Unique INTs:', len(set(ints_list)))
    print('Intro/Zero INTs:', ints_list.count(0))
    print('=' * 70)

    return ints_list


# Take as input array of integer and split it in chunks of equal length with padded values if necessary
def split_int_array_in_512chunks(array):

    chunks = [array[x:x + 512] for x in range(0, len(array), 512)]
    for i, c in enumerate(chunks):
        # Pad chunks shorter than 512 with 0
        if len(c) < 512:
            c = np.pad(c, (0, 512 - len(c) % 512), 'constant')
            c = c.tolist()
            chunks[i] = c

    return chunks

def split_list_on_value(l, value):
    size = len(l)
    idx_list = [idx for idx, val in
                enumerate(l) if val == 0]

    # idx_list = [idx for idx, val in
    #             enumerate(l) if val == 0 and l[idx + 1] == 127 + 128 and l[idx + 2] == 127 + 256]

    res = [l[i: j] for i, j in
           zip([0] + idx_list, idx_list +
               ([size] if idx_list[-1] != size else []))]

    res.pop(0)

    return res


# vector songs = song in int format
# can be divided in sub lists of max n tokens or a single list of int
def convert_vector_to_midi(token_song, ticks_per_quarter, track_title):
    for idx, out1 in enumerate(token_song):
        if len(out1) != 0:

            song = out1
            song_f = []
            time = 0
            dur = 0
            vel = 0
            pitch = 0
            channel = 0

            son = []

            song1 = []

            for s in song:
                if s > 127:
                    son.append(s)

                else:
                    if len(son) == 4:
                        song1.append(son)
                    son = []
                    son.append(s)

            for s in song1:
                if s[0] > 0 and s[1] >= 128:
                    if s[2] > 256 and s[3] > 384:
                        channel = s[0] // 11

                        vel = (s[0] % 11) * 19

                        time += (s[1] - 128) * 16

                        dur = (s[2] - 256) * 32

                        pitch = (s[3] - 384)

                        song_f.append(['note', time, dur, channel, pitch, vel])

            detailed_stats = TMIDIX.Tegridy_SONG_to_MIDI_Converter(song_f,
                                                                   output_signature='converted',
                                                                   output_file_name=r'converted_' + track_title,
                                                                   track_name=track_title,
                                                                   list_of_MIDI_patches=[0, 24, 32, 40, 42, 46, 56, 71, 73, 0, 53, 19, 0, 0, 0, 0],
                                                                   number_of_ticks_per_quarter=ticks_per_quarter)

            for key, value in detailed_stats.items():
                print('=' * 70)
                print(key, '|', value)

            print('Done!')
    return
