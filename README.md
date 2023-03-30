# Midi Classification and Generation 

## Project Structure

The project is divided in three main parts:

- Midi_processing (or utilities) 
- Classification 
- Generation.
   
The dataset folder has the three different MIDI databases containing the MIDI files employed in the project in zip file. 

### Midi Processing or Utilities

This section contains all the utilies useful for Midi processing. The main is the "manage_processing.py" file, from here the databases are unzipped and then processed to transform the MIDI files to the 'Melody Chords" and "Integer" format that will be used for the Classification and Generation. This process creates pickle files from the MIDI songs with their representation: one pickle including labels for the classification task, and one pickle without labels for the Generation task. Moreover csv file are created with all the representation obtained during the process for the three different MIDI databases.  


### Generation 

This section is composed of two different file, one for finetuning and one for the actual genearation. 
The finetuning file use as starting point an already trained model downlodable from https://github.com/asigalov61/Mini-Muse/tree/main/Model and present in the database folder. You can finetune the model on the different databases already processed and transformed. The generation file take the finetuned model and use that to generate original continuation of little MIDI extract from the databases.

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


### MIDI processing

Follow this steps to process the three database (Classic, NES, and Rock) and obtain the representation needed to procede in the next sections.

1 - Inside the project directories, go to the *Midi_processing* folder and run the *manage_processing.py* file, this processing will create the necessary files and will store the integer representations and csv file inside the dataset folder. 

Note that if you want to try only a dataset you can comment the lines of code related to the other databases in the *manage_processing.py* file. 



### Classification 

1 - Set *manage_classification.py* come main file to run with the correct interpreter from the virtualenv

2 - Inside the *manage_classification.py* file, in the main, you have to decomment the lines of code related to the database you want to train the model on (from default is set to the classic db)

3 - Run the *manage_classification.py* file and the training will start outputting in the classification folder the metrics results and in the models folder the trained model



### Generation 

Install required depencies from terminal
```console
pip install torch==1.12.1+cu116 -f https://download.pytorch.org/whl/cu116/torch_stable.html
```

The Generation is composed of two sections: 

- The finetuning section list the steps to reproduce the fine-tuning of the pre-trained model to create the model fine tuned on the database you want to choose (Classic, NES, or Rock)

- The actual generation section list the steps to use the fine tuned model from the previous section to generate the original song through the continuation of a song excerpts (You can execute only this section since in the project you have the already pre trained model on each dataset) 


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

2 - In project directories go inside the Generation folder and open the *finetuning.py* 

3 - Set the variable value at the beginiing of the file according to the model you want to finetune on:

- set *full_path_to_model_checkpoint* to the model path from the previous step
- set *ints_dataset* to the integer representation of the datbase you want to finetune on, this representation is created in the MIDI processing section inside the dataset folder (default set to rock dataset integer representation)
- set *dataset_test_path*, *path_to_best_checkpoint*, *loss_fig_path* (default set to rock dataset)

4 - Run the *finetuning.py* and fine tune the model

Note that the model was finetuned on a TESLA P6 GPU with 16 GB, if you run this section with a lower GPU is possible that the code give you an *out of memory* error. You can still try to get it running by lowering the values in the model configuration in the *finetuning.py* file. 



#### Actual generation 

This section is about the actual generation of the song from the fine tuned model. You can execute this phase without running any of the previous section by unzipping the already fine tuned models inside the *Model* folder in the *Generation* folder or you can use models fine tuned by you following the finetuning section instructions. 

1 - Set the parameters values in the *generation.py* file:

- *full_path_to_model_checkpoint* : path to the fine tuned model (default set to the NES dataset)

- *number_of_prime_tokens* : length in tokens of the MIDI excerpt to use to begin the generation  ( default set to 128)

- *number_of_continuation_blocks* : number of blocks of fixed length generated by the model as continuaion of the prime tokens (Defautl set to 1, note that increasing this value may or may not work well due to long-term structure decay) 

- *temperature* : number used to tune the degree of randomness of the generated tokens (default set to 0.5, play with this settings to get different results) 

- *song_timing* : song Timing when converted to audio (default set to 400)

- *priming_index* : index in the csv, created in the *Midi processing* section, of the song to use as priming for the generation (default set to 65)

2 - Once the parameters are set, run the *generation.py* file, it will output the generated song along with the original song and the excerpt all converted. Moreover it will create a *parameters.txt* file with the following information about the generated song:

- *Priming Song* : title of the song used as priming
- *Number of prime tokens* : number of prime tokens value
- *Priming index* : index of the priming song
- *Temperature* : temperature value
- *Number of continuation blocks* : number of continuation blocks value
- *Similarity distance normal and normalized*: Similarity value with the priming song in absolute value and normalize on the length of the generated song
- *Similarity Percentage* : similarity percentage value with the priming song
- *Original Genre of the priming song* : class of the priming song (depend on the dataset ex: in NES dataset can be one of rpg, sport, fighting, shooting, puzzle) 
- *Genres classified divided by chunks* : genre of the generated blocks classified with the trained model in classification section
- *Db Classified divided by chunks* : the generated blocks classified by the trained model in one of the three dataset (NES, Rock, or Classic)
- *Length of generated track in tokens* : length of the generated track


