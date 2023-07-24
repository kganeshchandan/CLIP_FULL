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
                  "batch_size":400,
                  "seq_len":70,
                  "splits":[0.8, 0.1, 0.1],
                  "num_workers":20
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
    'h_dim': 256,
    'max_time_steps': 1000,
    'num_heads': 7,
    'num_layers': 5,
    'output_size': 512,
    'patch_size': 7,
    'use_clf_token': True,
}

config['train'] = {
    'lr':0.0001,
    'temperature' :0.1,
    'checkpoint_dir': "checkpoints/temp",
    'device':"cuda",
    'num_epochs':100,
    'threshold': 0.9999,
    'weight_decay': 1.0e-06
}

config['wandb'] = {
    "dir": "/scratch/kanakala.ganesh/",
    "job_type": "sample",
    "project_name": "CLIP_Full_testing",
    "run_name": "RUN_testing"
}
config['data']['max_charge'] = None
config['data']['num_species'] = None

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
import wandb
import os 
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

from architecture_smiles import CLIP
from train_utils import CombinedLoss
from train_utils import train_clip, train_total, train_recon


logs, max_charge, num_species = None, None, None

def run(config):
    with wandb.init(project= config['wandb']['project_name'],
                    dir= config['wandb']['dir'],
                    name=config['wandb']['run_name'] ,
                    config = config,
                    job_type= config['wandb']['job_type'],
                    save_code= True):
        config = wandb.config
        global logs, max_charge, num_species
        num_gpus = torch.cuda.device_count()
        print("No of GPUs available", num_gpus)
        model = CLIP(config)
        model.to(device)
        model = torch.nn.parallel.DataParallel(model)
        
        optimizer = torch.optim.AdamW(model.parameters(), 
                                      lr = config['train']['lr'],
                                      weight_decay=config['train']['weight_decay'])
        vocab = pickle.load(open(config['data']['vocab_path'], 'rb'))
        loss_fn = CombinedLoss(vocab).to(device)
        
        logs = config['train']['logs']
        
        dataloaders, max_charge, num_species = prepare_data(config)
        for d in dataloaders:
            print("no of batches ", len(dataloaders[d]))
        
        config['data']['max_charge'] = max_charge
        config['data']['num_species'] = num_species
        
        print("Starting Training")
        
        wandb.watch(model, loss_fn, log='all', log_freq=100, log_graph=True)
        #train_clip(config, model, dataloaders, optimizer, loss_fn, logs, 0, 200)
        train_recon(config, model, dataloaders, optimizer, loss_fn, logs, 000, 200)
        train_total(config, model, dataloaders, optimizer, loss_fn, logs, 200,400)
run(config)
