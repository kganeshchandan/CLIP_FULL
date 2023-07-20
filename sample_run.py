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
                  "batch_size":128,
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
    'd_ff': 512,
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
    'temperature' :0.1
}


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

from PrepareData import prepare_data
dataloaders, max_charge, num_species = prepare_data(config)

from architecture import CLIP

model = CLIP(config)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr = 0.0001)
loss_fn = CombinedLoss().to(device)


train_losses = []
val_losses = []

for i in range(50):
    train_losses.append(train_one_epoch(model, dataloaders['train'], i, optimizer, loss_fn , focus="clip_loss"))
    val_losses.append(validate(model, dataloaders['test'], i, optimizer, loss_fn))
    # print("======================================")
    
for i in range(50):
    train_losses.append(train_one_epoch(model, dataloaders['train'], i, optimizer, loss_fn , focus="reconstruction_loss"))
    val_losses.append(validate(model, dataloaders['test'], i, optimizer, loss_fn))
    # print("======================================")
    
for i in range(50):
    train_losses.append(train_one_epoch(model, dataloaders['train'], i, optimizer, loss_fn , focus="total_loss"))
    val_losses.append(validate(model, dataloaders['test'], i, optimizer, loss_fn))
    
train_losses = np.array(train_losses)
val_losses = np.array(val_losses)

plt.plot(train_losses[:, 0], label="train_total_loss")
plt.plot(val_losses[:, 0],label="val_total_losses")
plt.title("Total loss")
plt.legend()
plt.savefig("total_loss.png")
plt.clf()

plt.plot(train_losses[:, 1], label="train_clip_loss")
plt.plot(val_losses[:, 1],label="val_clip_losses")
plt.title("clip loss")
plt.legend()
plt.savefig("clip_loss.png")
plt.clf()

plt.plot(train_losses[:, 2], label="total_recon_loss")
plt.plot(val_losses[:, 2],label="val_recon_losses")
plt.title("recon loss")
plt.legend()
plt.savefig("reconst_loss.png")


