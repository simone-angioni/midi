# Load the model and classify the song
# from tensorflow import keras
import numpy as np
import tensorflow as tf
from Midi_Processing.preprocess_data import convert_vector_to_midi

import sys

sys.path.append(
    r'C:\Users\andri\Desktop\Tesi\Midi_Classification_and_Generation\libraries\tegridy-tools\tegridy-tools')

import TMIDIX

from Midi_Processing.mini_muse_utils import split_int_array_in_512chunks

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
        model_path = r'..\Classification\models\runs_on_datset\nes\train_data2\transformer\folds\FullData'
        classes = NES_CATEGORIES
    if db_type == 2:
        model_path = r'..\Classification\models\runs_on_datset\rock\transformer\folds\FullData'
        classes = ROCK_CATEGORIES
    if db_type == 3:
        model_path = r'..\Classification\models\runs_on_datset\classic\transformer\folds\FullData'
        classes = CLASSIC_CATEGORIES

    model_intradb = load_model(model_path)
    model_interdb = load_model(r'..\Classification\models\runs_on_datset\inter_db\transformer\folds\FullData')

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

    int_song_path = r'..\converted_midi\song_generated\muse\droput ' \
                    r'0.3\nes\finetuning\batch_size4\gpt2_rpr_checkpoint_1_epoch_320000_steps_0.3397_loss.pth\3 vote ' \
                    r'6 of 10\int_representation '

    int_song = TMIDIX.Tegridy_Any_Pickle_File_Reader(int_song_path)

    classify_song(int_song, 1)
