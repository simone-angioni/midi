import numpy as np
import tensorflow as tf
from Midi_processing.preprocess_data import convert_vector_to_midi

import TMIDIX

from Midi_processing.mini_muse_utils import split_int_array_in_512chunks

NES_CATEGORIES = ["RPG", "Sport", "Fighting", "Shooting", "Puzzle"]
ROCK_CATEGORIES = ['Clapton', 'Queen', 'Beatles', 'Rolling Stones']
CLASSIC_CATEGORIES = ['Albanez', 'Beethoven', 'Mozart']
INTER_DB = ["Nes", "Rock", "Classic"]


def load_model(path):
    model = tf.keras.models.load_model(path)
    # print(model.summary())

    return model


# vector song = int tokens format
# path = path of the model to classify the song with
def classify_song(vector_song, db_type):

    if db_type == 1:
        model_path = r'..\Classification\models\NES_transformer_model\FullData'
        classes = NES_CATEGORIES
    if db_type == 2:
        model_path = r'..\Classification\models\Rock_transformer_model\FullData'
        classes = ROCK_CATEGORIES
    if db_type == 3:
        model_path = r'..\Classification\models\Classic_transformer_model\FullData'
        classes = CLASSIC_CATEGORIES

    model_intradb = load_model(model_path)
    model_interdb = load_model(r'..\Classification\models\InterDb_transformer_model\FullData')

    vector_song = np.array(vector_song)

    chunks = split_int_array_in_512chunks(vector_song)
    labels = []
    inter_db = []
    for c in chunks:
        c = np.array(c)
        c = c.reshape(1, 512)
        y = np.argmax(model_intradb.predict(c))
        labels.append(classes[y])

        y2 = np.argmax(model_interdb.predict(c))
        inter_db.append(INTER_DB[y2])

    return labels, inter_db


if __name__ == "__main__":

    # Try the classifier inserting path to int song representation that you want to classify
    int_song_path = r''

    int_song = TMIDIX.Tegridy_Any_Pickle_File_Reader(int_song_path)

    # Number 1 nes db - Number 2 Rock db - Number 3 Classic db
    classify_song(int_song, 1)
