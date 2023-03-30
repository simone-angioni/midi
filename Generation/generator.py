import sys

from pathlib import Path
path = Path(sys.path[0])
path_s = str(path.parent.absolute())
print("sys path:  " + path_s)
sys.path.append(path_s)

import itertools

import pandas as pd

import TMIDIX
from GPT2RGAX import *

from collections import OrderedDict

from Midi_processing.mini_muse_utils import convert_vector_to_midi

import copy

import json

from Midi_processing.find_similarity import calculate_similarity

from Classification.classifier import classify_song

def find_song_by_index(index, db):
    # The index respect to the csv visualization is - 2

    if db == 'nes':
        csv_path = path_s + '/dataset/nes/csv/nes_labelled2.csv'
        
    elif db == 'rock':
        csv_path = path_s + '/dataset/rock/csv/rock_labelled.csv'

    if db == 'classic':
        csv_path = path_s + '/dataset/classic/csv/classic_labelled.csv'

    df = pd.read_csv(csv_path, sep=';', converters={'int_tokens': eval})

    song = df.iloc[[index]]
    
    print("Title of the priming song: ", + song['filename'])

    return song


def config_model():
    DIC_SIZE = 512

    max_seq = 1024

    print('Loading the model...')
    config = GPTConfig(DIC_SIZE,
                       max_seq,
                       dim_feedforward=512,
                       n_layer=8,
                       n_head=8,
                       n_embd=512,
                       enable_rpr=True,
                       er_len=max_seq)

    return config


if __name__ == "__main__":

    # Model checkpoint fine-tuned on NES DB
    full_path_to_model_checkpoint = path_s + r'/Generation/models/NES_model.pth'

    # Model checkpoint fine-tuned on Rock DB
    # full_path_to_model_checkpoint = path_s + r'/Generation/models/Rock_model.pth'

    # Model checkpoint fine-tuned on Classic DB
    # full_path_to_model_checkpoint = path_s + r'/Generation/models/Classic_model.pth'

    config = config_model()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GPT(config)

    state_dict = torch.load(full_path_to_model_checkpoint, map_location=device)

    new_state_dict = OrderedDict()
    
    # This piece of code can be used if there is an error loading the model
    # for k, v in state_dict.items():
    #     name = k[7:]  # remove 'module'
    #     new_state_dict[name] = v

    model.load_state_dict(state_dict)

    model.to(device)

    model.eval()
    # print(model)

    # Parameters Setting

    # Generate Prime
    number_of_prime_tokens = 128

    # This may or may not work well due to long-term structure decay
    number_of_continuation_blocks = 1 

    # Play with the settings to get different results
    temperature = 0.5

    # Song Timing when converted
    song_timing = 400

    # Index in the csv of the song wanted to use as priming
    priming_index = 65
    # 999 = let it be

    print('Selected prime #', priming_index)
    # print('Prime index:', iindex)

    # Use the row based on the model that we are using to generate
    # prime_song = find_song_by_index(priming_index, 'rock')
    prime_song = find_song_by_index(priming_index, 'nes')
    # prime_song = find_song_by_index(priming_index, 'classic')

    print(prime_song)

    prime_song_tokens = prime_song['int_tokens'].values[0]
    prime_song_title = str(prime_song['filename'].values[0])

    # save to disk the original song from where the prime come
    prime_single_list = list(itertools.chain.from_iterable(prime_song_tokens))
    convert_vector_to_midi([prime_single_list], song_timing, prime_song_title)

    out1 = prime_song_tokens[0][:number_of_prime_tokens + 1]

    pr = 'prime_' + prime_song_title

    convert_vector_to_midi([out1], song_timing, pr)

    # Single Continuation Block Generator

    show_stats = False

    # ===================================================================
    print('Number of prime tokens:', number_of_prime_tokens)
    print('Model temperature:', temperature)

    print('=' * 70)
    print('Generating...')

    inputs = prime_song_tokens[0][:number_of_prime_tokens+1]

    rand_seq = model.generate(torch.Tensor(inputs),
                              target_seq_length=1024,
                              temperature=temperature,
                              stop_token=512,
                              verbose=show_stats)

    out1 = rand_seq[0].cpu().numpy().tolist()

    convert_vector_to_midi([out1], song_timing, 'continuation_block')

    # Auto-continue resulting composition

    out2 = copy.deepcopy(out1)

    for i in range(number_of_continuation_blocks):
        rand_seq = model.generate(torch.Tensor(out2[-64:]),
                                  target_seq_length=1024,
                                  temperature=temperature,
                                  stop_token=512,
                                  verbose=show_stats)

        out = rand_seq[0].cpu().numpy().tolist()

        out2.extend(out[64:])

    original_genre = prime_song['label'].values

    print("Similarity values min = ", str(number_of_prime_tokens * 2) + " max = 2050")
    dist, dist2 = calculate_similarity(out2, prime_song_tokens)

    # Report distance value to percentage of similarity with formula: ((input - min) * 100) / (max - min)
    # Min = 64 *2, max = 1984 * 2, input = dist2 returned from algorithm
    similarity_percentage = ((dist2 - number_of_prime_tokens*2) * 100) / ((len(out2)*2) - (number_of_prime_tokens*2))

    # Save to disk the original vector and its int representation
    TMIDIX.Tegridy_Any_Pickle_File_Writer(out2, 'int_representation')

    convert_vector_to_midi([out2], song_timing, 'final_composition')

    # Classify song with our trained model
    del model
    torch.cuda.empty_cache()
    #genres, classified_db = classify_song(out2, 1)

    # Create a dictionary with parameters value and similarity scores
    d = dict()

    d = {
        "Priming Song": prime_song_title,
        "Number of prime tokens": number_of_prime_tokens,
        "Priming index": priming_index,
        "Temperature": temperature,
        "Number of continuation blocks": number_of_continuation_blocks,
        "Similarity distance normal and normalized": str(dist2) + '----' + str(dist),
        "Similarity Percentage": similarity_percentage,
        # "Instruments used to generate": instrument_name,
        "Original Genre of the priming song": str(original_genre),
        #"Genres classified divided by chunks: ": str(genres),
        #"Db Classified divided by chunks": str(classified_db),
        "Length of generated track in tokens": len(out2)
    }

    # Write dictionary to text file
    with open('parameters.txt', 'w') as convert_file:
        convert_file.write(json.dumps(d))
