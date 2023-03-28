# Midi Classification and Generation 

## Project Structure
The project is divided in three main parts: Midi_processing (or utilities), Classification and Generation.   
The dataset folder has the three different MIDI databases containing the MIDI files employed in the project in zip file. 

### Midi Processing or Utilities
This section contains all the utilies useful for Midi processing. The main is the "manage_processing.py" file, from here the databases are unzipped and then processed to transform the MIDI files to the 'Melody Chords" and "Integer" format that will be used for the Classification and Generation. This process creates pickle files from the MIDI songs with their representation: one pickle including labels for the classification task, and one pickle without labels for the Generation task. Moreover csv file are created with all the representation obtained during the process for the three different MIDI databases.  

### Generation 
This section is composed of two different file, one for finetuning and one for genearation. The finetuning file use as starting point an already trained model downlodable from https://github.com/asigalov61/Mini-Muse/tree/main/Model and present in the database folder. You can finetune the model on the different databases already processed and transformed. The generation file take the finetuned model and use that to generate original continuation of little MIDI extract from the databases.

### Classification
The classification part face a multi-class classification problem, classifyng chunks of MIDI songs already processed based on the label they have. For instance, in the NES dataset we have five classes based on the genre of videogames of which the song is part. The genres are Role-Playing Games, Sport, Fighting, Shooting and Puzzle. To do so we train three classical machine learning baseline methods such as K-Neighbour, Random Forest and Support Vector Machine and a model based on the new Transformer technology in order to compare the results obtained from the different models. 

## Run Instructions

### Prerequisites 

Python verson 3.10 (downlodable from https://www.python.org/downloads/release/python-3100/)

CUDA drivers version 11.6 (downloadable from https://developer.nvidia.com/cuda-11-6-0-download-archive)

1 - Create virtual enviroment 'midi_enviroment' following tutorial on https://docs.python.org/3/library/venv.html

2 - Activate the virtual enviroment

3 - Install library requirements from terminal 
```console
pip install -r requirements.txt
```

### Classification 

1 - Set *manage_classification.py* come main file to run with the correct interpreter from the virtualenv

2 - Inside the *manage_classification.py* file you have to decomment the lines of code related to the database you want to train the model on (from default is set to the classic db)

3 - Run the *manage_classification.py* file and the training will start outputting in the classification folder the metrics results and in the models folder the trained model

### Generation 

Install required depencies from terminal
```console
torch==1.12.1+cu116 -f https://download.pytorch.org/whl/cu116/torch_stable.html
```

The Generation is composed of two sections: 
- The finetuning section list the steps to reproduce the fine-tuning of the pre-trained model to create the model fine tuned on the database you want to choose (Classic, NES, or Rock)
- The actual generation section list the steps to use the fine tuned model from the previous section to generate the original song through the continuation of a song excerpts 

#### Finetuning 

1 - Follow this steps to download the pre trained model: 

- Go to the page https://github.com/asigalov61/Mini-Muse/tree/main/Model 
- Download each one of the zip file inside the same folder
- Concatenate the splitted zip files into one with the following commands

On windows 
```console
copy /B Mini-Muse-Trained-Model.zip.0* model.zip
```

On Linux 
```console
cat Mini-Muse-Trained-Model.zip.0* > model.zip
```

- Extract the model with extension *pth* from the zip file

2 - In project directories go inside the Generation folder, open the *finetuning.py* file and set the variable *full_path_to_model_checkpoint* to the model path


