
from PrepareData import prepare_data
import torch
import pickle 
import wandb
import yaml
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

from architecture import CLIP
from train_utils import CombinedLoss
from train_utils import train_clip, train_total, train_recon


logs, max_charge, num_species = None, None, None
logs = {
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
        
        global logs
        
        dataloaders, max_charge, num_species = prepare_data(config)
        for d in dataloaders:
            print("no of batches ", len(dataloaders[d]))
        
        config['data']['max_charge'] = max_charge.item()
        config['data']['num_species'] = num_species
        
        print("Starting Training")
        
        wandb.watch(model, loss_fn, log='all', log_freq=100, log_graph=True)
        #train_clip(config, model, dataloaders, optimizer, loss_fn, logs, 0, 200)
        #train_recon(config, model, dataloaders, optimizer, loss_fn, logs, 200, 300)
        train_total(config, model, dataloaders, optimizer, loss_fn, logs, 000,500)
        

if __name__ == '__main__':
    config = yaml.safe_load(open(sys.argv[1], 'r'))
    run(config)
