
import torch
from torch import nn
import pickle
from qm9 import utils as qm9_utils
from models.vit import ViT
from qm9.models import EGNN

device = torch.device("cuda")
dtype = torch.float32

from models.decoder import LatentToMol

def set_up_causal_mask(seq_len):
    mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask.requires_grad = False
    return mask



import torch
from torch import nn
import pickle
from qm9 import utils as qm9_utils
from models.vit import ViT
from qm9.models import EGNN

device = torch.device("cuda")
dtype = torch.float32

from models.decoder import LatentToMol

def set_up_causal_mask(seq_len):
    mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask.requires_grad = False
    return mask

class PositionalEncodings(nn.Module):
    """Attention is All You Need positional encoding layer"""

    def __init__(self, seq_len, d_model, p_dropout):
        """Initializes the layer."""
        super(PositionalEncodings, self).__init__()
        token_positions = torch.arange(start=0, end=seq_len).view(-1, 1)
        dim_positions = torch.arange(start=0, end=d_model).view(1, -1)
        angles = token_positions / (10000 ** ((2 * dim_positions) / d_model))

        encodings = torch.zeros(1, seq_len, d_model)
        encodings[0, :, ::2] = torch.cos(angles[:, ::2])
        encodings[0, :, 1::2] = torch.sin(angles[:, 1::2])
        encodings.requires_grad = False
        self.register_buffer("positional_encodings", encodings)

        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x):
        """Performs forward pass of the module."""
        x = x + self.positional_encodings
        x = self.dropout(x)
        return x

    
class bottle(nn.Module):
    def __init__(self, seq_len, hidden_size):
        super(bottle, self).__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.down = nn.Sequential(
            nn.Linear(seq_len*hidden_size, 512),
            nn.LeakyReLU(),
            nn.Linear(512, hidden_size)
        )
        
    def forward(self, inp):
        inp = inp.view(-1, self.seq_len * self.hidden_size)
        embed = self.down(inp)
        # out = self.up(embed)
        return embed
       

class CLIP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab = pickle.load(open(config['data']['vocab_path'], 'rb'))
        self.temperature = config['train']['temperature']
        self.max_charge = config['data']['max_charge']
        self.num_species = config['data']['num_species']
        self.embed = nn.Embedding(len(self.vocab), config['molecule_decoder']['hidden_size'], padding_idx=self.vocab.pad_index)
        self.pe = PositionalEncodings(d_model=config['molecule_decoder']['hidden_size'], p_dropout=0.1, seq_len=config['data']['seq_len'])
        
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=config['molecule_decoder']['hidden_size'],
                                                               nhead=4,
                                                               dropout=0.1,
                                                               batch_first=True)
        
        self.trfmencoder = nn.TransformerEncoder(encoder_layer=transformer_encoder_layer,
                                                 num_layers=3)
        
        self.Spectra_Encoder = ViT(
            patch_size = self.config['spectra_encoder']['patch_size'], 
            num_layers = self.config['spectra_encoder']['num_layers'], 
            h_dim = self.config['spectra_encoder']['h_dim'], 
            num_heads = self.config['spectra_encoder']['num_heads'], 
            output_size = self.config['spectra_encoder']['output_size'], 
            d_ff=self.config['spectra_encoder']['d_ff'], 
            max_time_steps=self.config['spectra_encoder']['max_time_steps'], 
            use_clf_token=self.config['spectra_encoder']['use_clf_token'],
            dropout = self.config['spectra_encoder']['dropout'],
            dropout_emb = self.config['spectra_encoder']['dropout_emb']   
        )
        
        self.smiles_decoder = LatentToMol(
            in_size=self.config['molecule_decoder']['latent_size'],
            hidden_size=self.config['molecule_decoder']['hidden_size'], 
            n_layers=self.config['molecule_decoder']['n_layers'], 
            n_heads = self.config['molecule_decoder']['n_heads'],
            seq_len=self.config['data']['seq_len'], 
            vocab = self.vocab)
        
        self.bottle = bottle(config['data']['seq_len'], config['molecule_decoder']['hidden_size'])
        
        self.logit_scale = nn.Parameter(torch.ones([]) * self.temperature)
        
    
    def forward_mol(self, data):
        smi = data['decoder_inp'].to(device)
        smi = self.embed(smi)
        # smi = self.res_block(smi)
        smi = self.pe(smi)
        mem = self.trfmencoder(smi)
        mol_features = self.bottle(mem)
        
        mol_features = mol_features / mol_features.norm(dim=1, keepdim=True)
        
        return mol_features
    
    def forward_spec(self, data):
        spectra = data['IR'].to(device, dtype)
        spectra = torch.unsqueeze(spectra, 1)
        spectra = torch.unsqueeze(spectra, 1)
        
        spectra_features = self.Spectra_Encoder(spectra)
        spectra_features = spectra_features / spectra_features.norm(dim=1, keepdim=True)
        
        return spectra_features
    
    def forward_decoder(self, data, spec_latents):
        smi = data['decoder_inp'].to(device)
        tgt = data['decoder_tgt'].to(device)
        tgt_padding_mask = data['tgt_padding_mask'].to(device)
        tgt_mask = set_up_causal_mask(self.config['data']['seq_len']).to(device)
        
        pred = self.smiles_decoder(spec_latents,
                                   smi,
                                   tgt_mask,
                                   tgt_padding_mask)
        return pred
        
    def forward(self, data):
        logits_scale = self.logit_scale.exp()
        
        mol_latents = self.forward_mol(data)
        spec_latents = self.forward_spec(data)
        
        smile_preds = self.forward_decoder(data, spec_latents)
        
        return mol_latents, spec_latents, smile_preds, logits_scale, data['index'] 
        
        
        
        
        
        