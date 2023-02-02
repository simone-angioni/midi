# Midi Classification and Generation 

## Project Structure
The project is divided in three main parts: Midi_processing (or utilities), Classification and Generation. 
The dataset folder contains the three different MIDI databases in zip file employed in the project. 

### Midi Processing or Utilities
This section contains all the utilies useful for Midi processing. The main is the "manage_processing.py" file, from here the databases are unzipped and then processed to transform the MIDI files to other format that will be used for the Classification and Generation. This process creates pickle files from the MIDI songs: one pickle including labels for the classification task, and one pickle without labels for the Generation task. Moreover csv file are created with all the representation obtained during the process for the three different MIDI databases.

### Generation 
This section is composed of two different file, one for finetuning and one for genearation. Th finetuning file use as starting point an already trained model downlodable from https://github.com/asigalov61/Mini-Muse/tree/main/Model. You can finetune the model on the different databases already processed and transformed. The generation file take the finetuned model and use that to generate original continuation of little MIDI extract from the databases.

### Classification
The classification part face a multi-class classification problem, classifyng chunks of MIDI songs already processed based on the label they have. For instance, in the NES dataset we have five classes based on the genre of videogames of which the song is part. The genres are Role-Playing Games, Sport, Fighting, Shooting and Puzzle. To do so we train three classical machine learning baseline methods such as K-Neighbour, Random Forest and Support Vector Machine and a model based on the new Transformer technology in order to compare the results obtained from the different models. 

## Run Instructions
1 - Install library requirements from requirements.txt -> pip install requirements
