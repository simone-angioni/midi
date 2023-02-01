# Midi Classification and Generation 

## Project Structure
The project is divided in three main parts: Midi_processing (or utilities), Classification and Generation. 
The dataset folder contains the three different MIDI databases in zip file employed in the project. 

### Midi Processing or Utilities
This section contains all the utilies useful for Midi processing. The main is the "manage_processing.py" file, from here the databases are unzipped and then processed to transform the MIDI files to other format that will be used for the Classification and Generation. This process creates pickle files from the MIDI songs: one pickle including labels for the classification task, and one pickle without labels for the Generation task. Moreover csv file are created with all the representation obtained during the process for the three different MIDI databases.

### Generation 
This section is composed from two different file, one for the finetuning and one for the genearation. Th finetuning file use an already trained model downlodable from https://github.com/asigalov61/Mini-Muse/tree/main/Model

## Run Instructions
1 - Install library requirements from requirements.txt -> pip install requirements
