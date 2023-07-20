config = {}
config['data'] = {"qm9_broad_ir_path":'/home2/kanakala.ganesh/ir_data/qm9_broad_ir.pkl',
                  "vocab_path":'/home2/kanakala.ganesh/CLIP_PART_1/data/qm9_vocab.pkl',
                  "datafiles" : {
                        'train': '/home2/kanakala.ganesh/ir_data/raw_train.pickle',
                        'test':  '/home2/kanakala.ganesh/ir_data/raw_test.pickle',
                        'val':   '/home2/kanakala.ganesh/ir_data/raw_val.pickle'
                        },
                  "normalization" : "unit",
                  "shuffle": True,
                  "batch_size":100,
                  "seq_len":70,
                  "splits":[0.8, 0.1, 0.1],
                  "num_workers":16
                }

config['molecule_encoder'] = {
    'attention': 1,
    'coords_weight' :1.0,
    'device': "cuda",
    'hidden_nf':256,
    'in_edge_nf':0,
    'in_node_nf':15,
    'n_layers': 3,
    'node_attr': 1,
    'output_size':512
}

config['molecule_decoder'] = {
    'in_size': 512,
    'latent_size' : 512,
    'hidden_size': 512,
    'n_layers' : 5,
    'n_heads' : 4
}

config['spectra_encoder'] = {
    'd_ff': 1024,
    'dropout': 0.0,
    'dropout_emb': 0.1,
    'h_dim': 128,
    'max_time_steps': 1000,
    'num_heads': 5,
    'num_layers': 5,
    'output_size': 512,
    'patch_size': 7,
    'use_clf_token': True,
}

config['train'] = {
    'lr':0.00005,
    'temperature' :0.1,
    'checkpoint_dir': "checkpoints/temp",
    'device':"cuda",
    'num_epochs':100,
    'threshold': 0.99,
    'weight_decay': 1.0e-06
}

config['wandb'] = {
    "dir": "/scratch/kanakala.ganesh/",
    "job_type": "sample",
    "project_name": "CLIP_Full_testing",
    "run_name": "RUN_testing"
}
from PrepareData import prepare_data

import torch
from torch import nn, optim, Tensor
from torch.nn import functional as F
import pickle 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import seaborn as sns
import plotly

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

from architecture import CLIP

vocab = pickle.load(open( config['data']['vocab_path'], 'rb'))

class CombinedLoss(nn.Module):
    # under construction 
    def __init__(self, temperature=1, threshold=0.8):
        super().__init__()
        self.temperature = temperature
        self.threshold = threshold
    
    def forward(self, mol_features, spectra_features, logit_scale, smile_ypred, data):
        # spectra = spectra.squeeze(1)
        # spectra = spectra.squeeze(1)
        
        logits = logit_scale *  mol_features @ spectra_features.t() 
        
        mol_sims = mol_features @ mol_features.t()
        spectra_sims = spectra_features @ spectra_features.t()
        # og_spectra_sims = spectra @ spectra.t()
        
        # targets = get_spec_mat(spectra, threshold=self.threshold)
        targets = torch.diag(torch.ones(spectra_features.shape[0])).to(device)
      
        clip_loss = (F.cross_entropy(logits, targets) + 
                     F.cross_entropy(logits.t(), targets.t())
                     ) / 2
        
        smile_y = data['decoder_tgt'].to(device)
        smile_yprob = F.log_softmax(smile_ypred, dim=2)
        
        reconstruction_loss = F.nll_loss(smile_yprob.view(-1, len(vocab)),
                                        smile_y.view(-1))
        total_loss = clip_loss + reconstruction_loss
        
        return total_loss, clip_loss, reconstruction_loss
    
def train_one_epoch(model, dataloader, epoch, optimizer, loss_fn, focus="clip_loss"):
    
    running_loss = []
    model.to(device)
    model.train()
    
    for i, data in enumerate(dataloader):
        
        optimizer.zero_grad()
        
        mol_latents, spec_latents, smile_preds, logit_scale, ids = model(data, max_charge, num_species)
        total_loss, clip_loss, reconstruction_loss = loss_fn(mol_latents, spec_latents, logit_scale, smile_preds, data)
        
        if focus == "total_loss":
            total_loss.backward()
        elif focus == "clip_loss":
            clip_loss.backward()
        elif focus == "reconstruction_loss":
            reconstruction_loss.backward()
            
        optimizer.step()
        
        print( 'Training Epoch: {} | iteration: {}/{} | Loss: {}'.format(epoch, i, len(dataloader), total_loss.item() ), end='\r')
        running_loss.append([total_loss.item(), clip_loss.item(), reconstruction_loss.item()])
    
    running_loss = np.array(running_loss)
    return np.mean(running_loss, axis= 0)
        
def validate(model, dataloader, epoch, optimizer, loss_fn):
    
    running_loss = []
    model.to(device)
    model.eval()
    
    for i, data in enumerate(dataloader):    
        with torch.no_grad():
            mol_latents, spec_latents, smile_preds, logit_scale, ids = model(data, max_charge, num_species)
            total_loss, clip_loss, reconstruction_loss = loss_fn(mol_latents, spec_latents, logit_scale, smile_preds, data)
    
        print( 'Validation Epoch: {} | iteration: {}/{} | Loss: {}'.format(epoch, i, len(dataloader), total_loss.item() ), end='\r')
        running_loss.append([total_loss.item(), clip_loss.item(), reconstruction_loss.item()])
        
    running_loss = np.array(running_loss)
    return np.mean(running_loss, axis = 0)

config['train']['logs'] = {
            'train_total_loss':[],
            'train_clip_loss':[],
            'train_recon_loss':[],
            
            'val_total_loss':[],
            'val_clip_loss':[],
            'val_recon_loss':[],
            
            'test_total_loss':[],
            'test_clip_loss':[],
            'test_recon_loss':[],
            
            'best_epoch': -1,
            'best_clip_epoch': -1,
            'best_recon_epoch':-1,
            
            'best_total_loss':1000,
            'best_clip_loss':1000,
            'best_recon_loss':1000
        }   
 
import os
import yaml

def save_model(model, config, logs, name):
    
    path_dir = config['train']['checkpoint_dir']          
    if not os.path.exists(path_dir):
        os.mkdir(path_dir)
    model_path = path_dir + '/' + name + '.pth'
    config_path = path_dir + '/config.yaml'
    logs_path = path_dir + '/logs.pickle'
    
    torch.save(model.state_dict(), model_path)
    
    with open(config_path,'w') as yaml_file:
        yaml.dump(dict(config), yaml_file)
    with open(logs_path, 'wb') as file:
        pickle.dump(logs, file)
        
    print("Saved to {}".format(path_dir))
    
def load_model(path_to_dir):
    files = os.listdir(path_to_dir)
    for file in files:
        if '.pth' in file:      
            model_path = path_to_dir + '/' + file
        if '.yaml' in file:
            config_path = path_to_dir + '/' + file
    with open(config_path,'r') as f:
        config = yaml.full_load(f)
        
    model = CLIP(config)
    model.load_state_dict(torch.load(model_path))
    return model

def update_logs_and_checkpoints(model, tl, vl, epoch, logs):
    logs['train_total_loss'].append(tl[0])
    logs['train_clip_loss'].append(tl[1])
    logs['train_recon_loss'].append(tl[2])
    
    logs['val_total_loss'].append(vl[0])
    logs['val_clip_loss'].append(vl[1])
    logs['val_recon_loss'].append(vl[2])
    
    if vl[0] < logs['best_total_loss']:
        logs['best_total_loss'] = vl[0]
        logs['best_epoch'] = epoch
        save_model(model, config, logs, 'best_total')

    if vl[1] < logs['best_clip_loss']:
        logs['best_clip_loss'] = vl[1]
        logs['best_clip_epoch'] = epoch 
        save_model(model, config, logs, 'best_clip')
           
    if vl[2] < logs['best_recon_loss']:
        logs['best_recon_loss'] = vl[2]
        logs['best_recon_epoch'] = epoch
        save_model(model, config, logs, 'best_recon')
                
    return logs

def print_status(logs, time=None):
    train_total_loss = logs['train_total_loss'][-1]
    val_total_loss = logs['train_total_loss'][-1]
    print("Latest Train_Loss: {}, Latest Val_Loss: {}".format( train_total_loss, val_total_loss))
    print("Best Test_Loss: {}, Best Epoch: {}".format( logs['best_total_loss'],logs['best_epoch']))
    print("=============== Time: {}========================".format(time))
  
import wandb
import time
max_charge, num_species = None, None

def train_clip(model, dataloaders, optimizer, loss_fn, logs, num_epochs=50 ):
    for epoch in range(num_epochs):
        start = time.time()
        tl = train_one_epoch(model, dataloaders['train'], epoch, optimizer, loss_fn , focus="clip_loss")
        vl = validate(model, dataloaders['val'], epoch, optimizer, loss_fn )
        logs = update_logs_and_checkpoints(model, tl, vl, epoch, logs)
        end = time.time()
        
        wandb.log(
            {
                'epoch': epoch,
                'train_total_loss':tl[0],
                'train_clip_loss':tl[1],
                'train_recon_loss':tl[2],    
                'val_total_loss':vl[0],
                'val_clip_loss':vl[1],
                'val_recon_loss':vl[2],
            },
            step = epoch
        )
        clip_performance(model, dataloaders, epoch)
        print_status(logs, end-start)
    return logs

def train_recon(model, dataloaders, optimizer, loss_fn, logs, num_epochs=50 ):
    for epoch in range(num_epochs):
        start = time.time()
        tl = train_one_epoch(model, dataloaders['train'], epoch, optimizer, loss_fn , focus="reconstruction_loss")
        vl = validate(model, dataloaders['val'], epoch, optimizer, loss_fn )
        logs = update_logs_and_checkpoints(model, tl, vl, epoch, logs)
        end = time.time()
        
        wandb.log(
            {
                'epoch': epoch,
                'train_total_loss':tl[0],
                'train_clip_loss':tl[1],
                'train_recon_loss':tl[2],    
                'val_total_loss':vl[0],
                'val_clip_loss':vl[1],
                'val_recon_loss':vl[2],
            },
            step = epoch
        )
        print_status(logs, end-start)
    return logs

def train_total(model, dataloaders, optimizer, loss_fn, logs, num_epochs=50 ):
    for epoch in range(num_epochs):
        start = time.time()
        tl = train_one_epoch(model, dataloaders['train'], epoch, optimizer, loss_fn , focus="total_loss")
        vl = validate(model, dataloaders['val'], epoch, optimizer, loss_fn )
        logs = update_logs_and_checkpoints(model, tl, vl, epoch, logs)
        end = time.time()
        
        wandb.log(
            {
                'epoch': epoch,
                'train_total_loss':tl[0],
                'train_clip_loss':tl[1],
                'train_recon_loss':tl[2],    
                'val_total_loss':vl[0],
                'val_clip_loss':vl[1],
                'val_recon_loss':vl[2],
            },
            step = epoch
        )
        print_status(logs, end-start)
    return logs

def top_scores(mat1, mat2, offset):
    """
    mat1 is [testset]
    mat2 is  [testset + trainset]
    
    """
    hits = []
    tops = [1,2,3,4,5,10]
    score = [0] * (len(tops))

    sims = mat1 @ mat2.t() # sims shape is [test_size, test+train_size]
    for k in range(len(tops)):
        for i, row in enumerate(sims):
            max_sims, ids = torch.topk(row, tops[k])
            if i in ids:
                score[k] += 1
        score[k] = score[k] / len(row)
        # break
    return np.array(tops), np.array(score)

# %matplotlib inline
def distance_distribution(molmat, specmat):
    sims = molmat @ specmat.t()
    diagonals = torch.diagonal(sims, 0).detach().numpy()
    sims = np.random.choice(sims.view(-1).detach().numpy(), len(diagonals))
    vals = np.concatenate((sims, diagonals), axis=0)
    pairs = ["pairs"] * len(diagonals)
    nonpairs = ["others"] * len(sims)
    df = pd.DataFrame()
    df['distance'] = vals
    df['labels'] = pairs + nonpairs
    sns.histplot(df, x='distance', hue='labels', kde=True, bins=50)
     
    return plt

from tqdm import tqdm

def clip_performance(model, dataloaders, epoch):
    model.eval()
    molembeds = []
    specembeds = []
    for i, data in tqdm(enumerate(dataloaders['test'])):    
        with torch.no_grad():
            mol_latents, spec_latents, smile_preds, logit_scale, ids = model(data, max_charge, num_species)
            molembeds.append(mol_latents)
            specembeds.append(spec_latents)
    test_molembeds = torch.cat(molembeds, 0)
    test_specembeds = torch.cat(specembeds, 0)
    
    molembeds = []
    specembeds = []
    for i, data in tqdm(enumerate(dataloaders['train'])):    
        with torch.no_grad():
            mol_latents, spec_latents, smile_preds, logit_scale, ids = model(data, max_charge, num_species)
            molembeds.append(mol_latents)
            specembeds.append(spec_latents)
    train_molembeds = torch.cat(molembeds, 0)
    train_specembeds = torch.cat(specembeds, 0)
    
    all_molembeds = torch.cat(( test_molembeds, train_molembeds), axis = 0)
    
    all_molembeds = all_molembeds.to("cpu")
    train_molembeds = train_molembeds.to("cpu")
    train_specembeds = train_specembeds.to("cpu")
    test_molembeds = test_molembeds.to("cpu")
    test_specembeds = test_specembeds.to("cpu")
    
    tops, scores, hits = top_scores(test_specembeds, all_molembeds)
    for k, acc in zip(tops, scores):
        # print("Full Top {} Spec".format(k), acc)
        wandb.log({"Full Top {} Spec".format(k): acc}, step=epoch)
    
    tops, scores, hits = top_scores(test_specembeds, test_molembeds )
    for k, acc in zip(tops, scores):
        # print("Test Top {} Spec".format(k), acc)
        wandb.log({"Test Top {} Spec".format(k): acc}, step=epoch)

    wandb.log({'Distance Distribution Train': distance_distribution(train_molembeds, train_specembeds)}, step=epoch) 
    wandb.log({'Distance Distribution Test': distance_distribution(test_molembeds, test_specembeds)}, step=epoch) 

       
    

logs = {}

def run(config):
    with wandb.init(project= config['wandb']['project_name'],
                    dir= config['wandb']['dir'],
                    name=config['wandb']['run_name'] ,
                    config = config,
                    job_type= config['wandb']['job_type'],
                    save_code= True):
        config = wandb.config
        global logs, max_charge, num_species
        model = CLIP(config)
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), config['train']['lr'] )
        loss_fn = CombinedLoss().to(device)
        
        logs = config['train']['logs']
        
        dataloaders, max_charge, num_species = prepare_data(config)
        
        print("Starting Training")
        
        wandb.watch(model, loss_fn, log='all', log_freq=100, log_graph=True)
        train_clip(model, dataloaders, optimizer, loss_fn, logs, 100)
        train_recon(model, dataloaders, optimizer, loss_fn, logs,  50)
        train_total(model, dataloaders, optimizer, loss_fn, logs, 50)
        
        
config['data']['batch_size'] = 64
run(config)
  
        