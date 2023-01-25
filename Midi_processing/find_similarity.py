# This file intend to find the similarity between two midi file
# Computing the cosine similarity between the melody chords representation obtained from the pre-processing phase
# Melody chords format = [ [time, duration, pitch, channel, velocity] , [..], [..], ...]

import os
import pandas as pd

import math

import itertools


from seq_alignment import local_similarity

import time

def calculate_similarity(generated_song, v_priming_song):
    # Join chunks from original priming song
    priming_song = list(itertools.chain.from_iterable(v_priming_song))

    start_time = time.time()

    sim_init = local_similarity(generated_song, priming_song, False)
    sim_init.run()

    # scores divided by the log of the song to confront length plus one
    # to compensate for varying piece length
    dist = sim_init.match_distance / ((math.log(len(generated_song))) + 1)

    dist2 = sim_init.match_distance

    print("The Value of similarity distance between the two track is (higher means more similarity): ", + dist)
    print("--- %s seconds to calculate ---" % (time.time() - start_time))
    print("---"*70)

    return dist, dist2
