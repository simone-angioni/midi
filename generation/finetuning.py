#activate virtualenv
!source Music_generation_env/bin/activate

!nvidia-smi

import os
from tqdm import tqdm
import random
import sys

print('Loading TMIDIX module...')


sys.path.append(r'/home/server/studenti/musicGeneration/Music_generation_env/tegridy-tools/tegridy-tools')
sys.path.append(r'/home/server/studenti/musicGeneration/Music_generation_env/lib/python3.7/site-packages')
import TMIDIX


from GPT2RGAX import *

from torchsummary import summary
from sklearn import metrics
import matplotlib.pyplot as plt

# Path parameters
full_path_to_model_checkpoint = "/home/server/studenti/musicGeneration/models/muse/PreTrained/Mini_Muse_Trained_Model_88000_steps_0.6129_loss.pth"
ints_dataset = '/home/server/studenti/musicGeneration/dataset/muse/rock/ints_rock_dataset.pickle'
dataset_test_path = '/home/server/studenti/musicGeneration/dataset/muse/rock/rock_dataset_test'
path_to_best_checkpoint = '/home/server/studenti/musicGeneration/models/muse/rock/best_accuracy.pth'
loss_fig_path = '/home/server/studenti/musicGeneration/models/muse/rock/best_loss_image.png'

"""## Load data already processed"""

DIC_SIZE = 512
max_seq = 1024

config = GPTConfig(DIC_SIZE, 
                   max_seq,
                   dim_feedforward=512,
                   n_layer=8, 
                   n_head=8,
                   n_embd=512,
                   enable_rpr=True,
                   er_len=max_seq)

# DO NOT FORGET TO ADJUST MODEL PARAMS IN GPT2RGAX module to your specs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GPT(config)

#model = nn.DataParallel(model)

state_dict = torch.load(full_path_to_model_checkpoint, map_location=device)

model.to(device)

#summary(model)

print(model.eval())

#cos_sim = metrics.pairwise.cosine_similarity(
#    model.tok_emb.weight.detach().cpu().numpy()
#)
#plt.figure(figsize=(8, 8))
#plt.imshow(cos_sim, cmap="inferno", interpolation="none")
#im_ratio = cos_sim.shape[0] / cos_sim.shape[1]
#plt.colorbar(fraction=0.046 * im_ratio, pad=0.04)
#plt.xlabel("Position")
#plt.ylabel("Position")
#plt.tight_layout()
#plt.plot()
#plt.savefig("/content/Euterpe-Positional-Embeddings-Plot.png", bbox_inches="tight")

!pip3 install pickle5
import pickle5 as pickle
with open(ints_dataset, "rb") as fh:
    train_data1 = pickle.load(fh)

# Show some ints representation stats
        
print('Total INTs:', len(train_data1))
print('Minimum INT:', min(train_data1))
print('Maximum INT:', max(train_data1))
print('Unique INTs:', len(set(train_data1)))
print('Intro/Zero INTs:', train_data1.count(0))
print('=' * 70)

"""## Test the processed dataset"""

print('Sample INTs', train_data1[:15])

out = train_data1[:16000]

if len(out) != 0:
    
    song = out
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

                vel = (s[0] % 10) * 19

                time += (s[1]-128) * 16

                dur = (s[2] - 256) * 32

                pitch = (s[3] - 384)
                                  
                song_f.append(['note', time, dur, channel, pitch, vel ])

    detailed_stats = TMIDIX.Tegridy_SONG_to_MIDI_Converter(song_f,
                                                        output_signature = 'Mini Muse',  
                                                        output_file_name = dataset_test_path, 
                                                        track_name='Data test',
                                                        list_of_MIDI_patches=[0, 24, 32, 40, 42, 46, 56, 71, 73, 0, 53, 19, 0, 0, 0, 0],
                                                        number_of_ticks_per_quarter=400)

    print('Done!')

"""## Training """

import secrets

#max_seq = 512

SEQ_LEN = max_seq

BATCH_SIZE = 4 # Change this to your specs
#BATCH_SIZE = 8

# DO NOT FORGET TO ADJUST MODEL PARAMS IN GPT2RGAX module to your specs

print('=' * 50)
print('Loading training data...')

data_train, data_val = torch.LongTensor(train_data1), torch.LongTensor(train_data1)

class MusicSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):

        rand = secrets.randbelow((self.data.size(0)-(self.seq_len+1)) // (self.seq_len+1)) * (self.seq_len+1)

        x = self.data[rand: rand + self.seq_len].long()
        trg = self.data[(rand+1): (rand+1) + self.seq_len].long()
        
        return x, trg

    def __len__(self):
        return self.data.size(0)

train_dataset = MusicSamplerDataset(data_train, SEQ_LEN)
val_dataset   = MusicSamplerDataset(data_val, SEQ_LEN)
train_loader  = DataLoader(train_dataset, batch_size = BATCH_SIZE)
val_loader    = DataLoader(val_dataset, batch_size = BATCH_SIZE)
print('=' * 50)
print('Total INTs in the dataset', len(train_data1))
print('Total unique INTs in the dataset', len(set(train_data1)))
print('Max INT in the dataset', max(train_data1))
print('Min INT in the dataset', min(train_data1))
print('=' * 50)
print('Max sequence lenght:', max_seq)
print('Length of the dataset:',len(train_dataset))
print('Number of batched samples per epoch:', len(train_data1) // max_seq // BATCH_SIZE)
print('=' * 50)
print('Sample train dataset:', train_dataset[0])
print('Sample val dataset:', val_dataset[0])
print('=' * 50)
print('Train loader length:', len(train_loader))
print('Val loader length:', len(val_loader))
print('=' * 50)
print('Done! Enjoy! :)')
print('=' * 50)

# DO NOT FORGET TO ADJUST MODEL PARAMS IN GPT2RGAX module to your specs

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#model = GPT(config)

#model = nn.DataParallel(model)

#model.to(device)

#=====

init_step = 0
lr = LR_DEFAULT_START
lr_stepper = LrStepTracker(d_model, SCHEDULER_WARMUP_STEPS, init_step)
eval_loss_func = nn.CrossEntropyLoss(ignore_index=DIC_SIZE)
train_loss_func = eval_loss_func

opt = Adam(model.parameters(), lr=lr, betas=(ADAM_BETA_1, ADAM_BETA_2), eps=ADAM_EPSILON)
lr_scheduler = LambdaLR(opt, lr_stepper.step)


#===

best_eval_acc        = 0.0
best_eval_acc_epoch  = -1
best_eval_loss       = float("inf")
best_eval_loss_epoch = -1
best_acc_file = path_to_best_checkpoint
best_loss_file = path_to_best_checkpoint
loss_train, loss_val, acc_val = [], [], []

for epoch in range(0, 2):
    new_best = False
    
    loss = train(epoch+1, 
                 model, train_loader, 
                 train_loss_func, 
                 opt, 
                 lr_scheduler, 
                 num_iters=-1, 
                 save_checkpoint_steps=8000)
    
    loss_train.append(loss)
    
    eval_loss, eval_acc = eval_model(model, val_loader, eval_loss_func, num_iters=-1)
    loss_val.append(eval_loss)
    acc_val.append(eval_acc)
    
    if(eval_acc > best_eval_acc):
        best_eval_acc = eval_acc
        best_eval_acc_epoch  = epoch+1
        torch.save(model.state_dict(), best_acc_file)
        new_best = True

    if(eval_loss < best_eval_loss):
        best_eval_loss       = eval_loss
        best_eval_loss_epoch = epoch+1
        torch.save(model.state_dict(), best_loss_file)
        new_best = True
    
    if(new_best):
        print("Best eval acc epoch:", best_eval_acc_epoch)
        print("Best eval acc:", best_eval_acc)
        print("")
        print("Best eval loss epoch:", best_eval_loss_epoch)
        print("Best eval loss:", best_eval_loss)

"""### Eval funct to eval separately if needed"""

init_step = 0
lr = LR_DEFAULT_START
lr_stepper = LrStepTracker(d_model, SCHEDULER_WARMUP_STEPS, init_step)
eval_loss_func = nn.CrossEntropyLoss(ignore_index=DIC_SIZE)
train_loss_func = eval_loss_func

opt = Adam(model.parameters(), lr=lr, betas=(ADAM_BETA_1, ADAM_BETA_2), eps=ADAM_EPSILON)
lr_scheduler = LambdaLR(opt, lr_stepper.step)


eval_loss, eval_acc = eval_model(model, val_loader, eval_loss_func, num_iters=-1)

"""## Save the model"""

print('Saving the model...')
torch.save(model.state_dict(), full_path_to_model_checkpoint)
print('Done!')

#Plot resulting training loss graph

tr_loss_list = [item for sublist in loss_train for item in sublist]
plt.plot([i for i in range(len(tr_loss_list))] ,tr_loss_list, 'b')
plt.savefig(loss_fig_path)



