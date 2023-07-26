
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

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

import os
import yaml

import wandb
import time




class CombinedLoss(nn.Module):
    # under construction 
    def __init__(self, vocab, temperature=1, threshold=0.8):
        super().__init__()
        self.temperature = temperature
        self.threshold = threshold
        self.vocab = vocab
    
    def forward(self, mol_features, spectra_features, logit_scale, smile_ypred, data):
        # spectra = spectra.squeeze(1)
        # spectra = spectra.squeeze(1)
        # print(logit_scale)
        # print(mol_features.shape)
        # print(spectra_features.shape)
        
        logits = logit_scale[0] *  mol_features @ spectra_features.t() 
        
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
        
        reconstruction_loss = F.nll_loss(smile_yprob.view(-1, len(self.vocab)),
                                        smile_y.view(-1))
        total_loss = clip_loss + reconstruction_loss
        
        return total_loss, clip_loss, reconstruction_loss

def train_one_epoch( config, model, dataloader, epoch, optimizer, loss_fn, focus="clip_loss"):
    
    running_loss = []
    model.to(device)
    model.train()
    max_charge = config['data']['max_charge']
    num_species = config['data']['num_species']
    for i, data in enumerate(dataloader):
        
        optimizer.zero_grad()
        data = {k: v.to(device) for k, v in data.items()}
        
        mol_latents, spec_latents, smile_preds, logit_scale, ids = model(data)
        total_loss, clip_loss, reconstruction_loss = loss_fn(mol_latents, spec_latents, logit_scale, smile_preds, data)
        
        if focus == "total_loss":
            total_loss.backward()
        elif focus == "clip_loss":
            clip_loss.backward()
        elif focus == "reconstruction_loss":
            reconstruction_loss.backward()
            
        optimizer.step()
        nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)

        print( 'Training Epoch: {} | iteration: {}/{} | Loss: {}'.format(epoch, i, len(dataloader), total_loss.item() ), end='\r')
        running_loss.append([total_loss.detach().item(), clip_loss.detach().item(), reconstruction_loss.detach().item()])
        del total_loss, clip_loss, reconstruction_loss, mol_latents, spec_latents, smile_preds, logit_scale
        
    running_loss = np.array(running_loss)
    return np.mean(running_loss, axis= 0)

def validate(config, model, dataloader, epoch, optimizer, loss_fn):
    
    running_loss = []
    # model.to(device)
    model.eval()
    max_charge = config['data']['max_charge']
    num_species = config['data']['num_species']
    
    # with torch.no_grad():
    for i, data in enumerate(dataloader):      
        data = {k: v.to(device) for k, v in data.items()}
        mol_latents, spec_latents, smile_preds, logit_scale, ids = model(data)
        total_loss, clip_loss, reconstruction_loss = loss_fn(mol_latents, spec_latents, logit_scale, smile_preds, data)
    
        print( 'Validation Epoch: {} | iteration: {}/{} | Loss: {}'.format(epoch, i, len(dataloader), total_loss.detach().item() ), end='\r')
        running_loss.append([total_loss.detach().item(), clip_loss.detach().item(), reconstruction_loss.detach().item()])
        del total_loss, clip_loss, reconstruction_loss, mol_latents, spec_latents, smile_preds, logit_scale
        
    running_loss = np.array(running_loss)
    plt.clf()
    df = pd.DataFrame(running_loss, columns=['ttl_loss', 'clip_loss', 'recon_loss'])
    plot = sns.histplot(df, kde=True, bins=50)
    wandb.log({"Validation Loss Distribution":wandb.Image(plot)}, step=epoch)
    plt.clf()
    del plot
    model.train()
    return np.mean(running_loss, axis = 0)


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

def update_logs_and_checkpoints(config, model, tl, vl, epoch, logs):
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
              
    for key in logs:
        if not isinstance(logs[key], list):
            wandb.log({key:logs[key]}, step=epoch)  
    return logs

def print_status(logs, time=None):
    train_total_loss = logs['train_total_loss'][-1]
    val_total_loss = logs['train_total_loss'][-1]
    print("Latest Train_Loss: {}, Latest Val_Loss: {}".format( train_total_loss, val_total_loss))
    print("Best Test_Loss: {}, Best Epoch: {}".format( logs['best_total_loss'],logs['best_epoch']))
    print("=============== Time: {}========================".format(time))
  
def train_clip(config, model, dataloaders, optimizer, loss_fn, logs, start=0, num_epochs=50 ):
    for epoch in range(start,num_epochs):
        start = time.time()
        tl = train_one_epoch(config, model, dataloaders['train'], epoch, optimizer, loss_fn , focus="clip_loss")
        vl = validate(config, model, dataloaders['val'], epoch, optimizer, loss_fn )
        logs = update_logs_and_checkpoints(config, model, tl, vl, epoch, logs)
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
        if epoch % 10 == 0:
            clip_performance(config, model, dataloaders, epoch)
        print_status(logs, end-start)
    return logs

def train_recon(config, model, dataloaders, optimizer, loss_fn, logs, start=0, num_epochs=50 ):
    for epoch in range(start, num_epochs):
        start = time.time()
        tl = train_one_epoch(config, model, dataloaders['train'], epoch, optimizer, loss_fn , focus="reconstruction_loss")
        vl = validate(config, model, dataloaders['val'], epoch, optimizer, loss_fn )
        logs = update_logs_and_checkpoints(config, model, tl, vl, epoch, logs)
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
        if epoch % 10 == 0:
            clip_performance(config, model, dataloaders, epoch)
        print_status(logs, end-start)
    return logs

def train_total(config, model, dataloaders, optimizer, loss_fn, logs, start=0, num_epochs=50 ):
    for epoch in range(start, num_epochs):
        start = time.time()
        tl = train_one_epoch(config, model, dataloaders['train'], epoch, optimizer, loss_fn , focus="total_loss")
        vl = validate(config, model, dataloaders['val'], epoch, optimizer, loss_fn )
        logs = update_logs_and_checkpoints(config, model, tl, vl, epoch, logs)
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
        if epoch % 10 == 0:
            clip_performance(config, model, dataloaders, epoch)
        print_status(logs, end-start)
    return logs

def top_scores(mat1, mat2):
    """
    mat1 is the first mat
    mat2 is the second mat
    d = []
    for i in mat1:
        for j in mat2:
            closest between i and j
            d.append(rank)
    """
    hits = []
    tops = [1,2,3,4,5, 7, 10]
    score = [0] * (len(tops))

    for i in range(mat1.shape[0]):
        sims = mat2 @ mat1[i].t()
        for k in range(len(tops)):
            max_sims, ids = torch.topk(sims, tops[k])
            if (i) in ids:
                score[k] += 1

    for i in range(len(tops)):
        score[i] = score[i] / mat1.shape[0]

    return np.array(tops), np.array(score ) * 100

# %matplotlib inline
def distance_distribution(molmat, specmat):
    sims = molmat @ specmat.t()
    diagonals = torch.diagonal(sims, 0).cpu().numpy()
    sims = np.random.choice(sims.view(-1).cpu().numpy(), len(diagonals))
    vals = np.concatenate((sims, diagonals), axis=0)
    pairs = ["pairs"] * len(diagonals)
    nonpairs = ["others"] * len(sims)
    df = pd.DataFrame()
    df['distance'] = vals
    df['labels'] = pairs + nonpairs
    plot = sns.histplot(df, x='distance', hue='labels', kde=True, bins=50)
    del sims, diagonals,  vals, df
    return plot

def distance_mat(molmat, specmat):
    sims = specmat[:1000].cpu() @ molmat.t()[:,:1000].cpu()
    plt.clf()
    img = sns.heatmap(data=sims, annot=None)
    del sims
    return img

def decoder_performance(config, model, dataloaders, epoch):
    model.eval()
    vocab = pickle.load(open(config['data']['vocab_path'], 'rb'))
    acc = []
    validity = []
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloaders['val'])):
            data = {k: v.to(device) for k, v in data.items()}
            smi = data['decoder_inp']
            y = data['decoder_tgt']
            _, spec, pred, _, _ = model(data)
            # spec = model.forward_spec(data)
            # pred = model.forward_decoder(data, spec)
            y_pred = torch.argmax(pred, dim=2)
            equal_count= 0
            validity_count = 0
            for i in range(spec.shape[0]):
                og_smile = ""
                for char in vocab.from_seq(y[i]):
                    if char != "<pad>" and char != "<eos>":
                        og_smile += char
                pred_smile = ""
                for char in vocab.from_seq(y_pred[i]):
                    if char != "<pad>" and char != "<eos>":
                        pred_smile += char
                if og_smile == pred_smile:
                    equal_count  += 1
                m = Chem.MolFromSmiles(pred_smile)
                if m is not None:
                    validity_count += 1
            acc.append(equal_count / spec.shape[0])
            validity.append(validity_count / spec.shape[0])
        print("Acc :",np.array(acc).mean(), " validity: ", np.array(validity).mean())
    # wandb.log({ "accuracy" : np.array(acc).mean() }, step=epoch)  
    # wandb.log({"validity":np.array(validity).mean()}, step=epoch) 
    return np.array(acc).mean(), np.array(validity).mean()
        

def clip_performance(config, model, dataloaders, epoch):
    # model.to(device)
    model.eval()
    max_charge = config['data']['max_charge']
    num_species = config['data']['num_species']

    molembeds = []
    specembeds = []
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloaders['val'])):    
            data = {k: v.to(device) for k, v in data.items()}
            mol_latents, spec_latents, smile_preds, logit_scale, ids = model(data)
            molembeds.append(mol_latents.detach().cpu())
            specembeds.append(spec_latents.detach().cpu())
        del mol_latents, spec_latents, smile_preds, logit_scale, ids

    test_molembeds = torch.cat(molembeds, 0)
    test_specembeds = torch.cat(specembeds, 0)
    
    molembeds = []
    specembeds = []
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloaders['train'])):    
            data = {k: v.to(device) for k, v in data.items()}
            mol_latents, spec_latents, smile_preds, logit_scale, ids = model(data)
            molembeds.append(mol_latents.detach().cpu())
                # specembeds.append(spec_latents.detach().cpu())
        del mol_latents, spec_latents, smile_preds, logit_scale, ids
    
    train_molembeds = torch.cat(molembeds, 0)
    # train_specembeds = torch.cat(specembeds, 0)
    
    all_molembeds = torch.cat(( test_molembeds, train_molembeds), axis = 0)
    del train_molembeds
    
    tops, scores = top_scores(test_specembeds, all_molembeds)
    del all_molembeds
    
    for k, acc in zip(tops, scores):
        # print("Full Top {} Spec".format(k), acc)
        wandb.log({"Full Top {} Spec".format(k): acc}, step=epoch)
        

    tops, scores = top_scores(test_specembeds, test_molembeds )
    for k, acc in zip(tops, scores):
        # print("Test Top {} Spec".format(k), acc)
        wandb.log({"Test Top {} Spec".format(k): acc}, step=epoch)

    # wandb.log({'Distance Distribution Train': distance_distribution(train_molembeds, train_specembeds)}, step=epoch) 
    # del train_molembeds, train_specembeds
    wandb.log({'Distance Distribution Test': wandb.Image(distance_distribution(test_molembeds, test_specembeds))}, step=epoch) 
    plt.clf()
    wandb.log({'Similarity Matrix Test':wandb.Image(distance_mat(test_specembeds, test_molembeds))})
    plt.clf()
    decoder_acc, decoder_validity = decoder_performance(config, model, dataloaders, epoch)
    wandb.log({ "accuracy" : decoder_acc }, step=epoch)  
    wandb.log({"validity":decoder_validity}, step=epoch)
    del test_molembeds, test_specembeds
    model.train()
    plt.clf()
