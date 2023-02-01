# midi

##Project Structure
The project is divided in three main parts: Midi_processing (or utilities), Classification and Generation.
The dataset folder contains the three different MIDI databases employed in the project.

###Midi Processing or Utilities
In this section the main is the "manage_processing.py" file, from where the database are unzipped and then processed to transform the MIDI files to other format that will be used in Classification and Generation. This process create pickle files for the MIDI songs: one pickle including labels for the classification task, and one pickle without for the Generation task. Moreover csv file are created with all the representation obtained during the process for the three different MIDI databases.   

## Run Instructions
1 - Install library requirements from requirements.txt -> pip install requirements
