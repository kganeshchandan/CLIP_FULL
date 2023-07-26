import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F

def set_up_causal_mask(seq_len):
    mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask.requires_grad = False
    return mask
    
# class LatentEncoder(nn.Module):
#     def __init__(self, in_size, hidden_size, dropout=0.1):
#         super(LatentEncoder, self).__init__()
#         self.in_size = in_size

#         self.featurizer = nn.Sequential(
#             ResidualBlock(in_size),
#             nn.LeakyReLU(),
#             nn.Conv1d(in_channels=1,out_channels=5, kernel_size=3),
#             nn.MaxPool1d(2),
#             nn.BatchNorm1d(5),
#             nn.Conv1d(in_channels=5, out_channels=hidden_size,kernel_size=7),
#         )
    
#     def forward(self, x):
#         x = x.unsqueeze(1)
#         x = self.featurizer(x)
#         x = x.permute(0,2,1)
#         # returns batch, no_of_words, word_emb 
#         return x
 
class LatentEncoder(nn.Module):
    def __init__(self, in_size, seq_len =70, hidden_size=512, dropout=0.1):
        super(LatentEncoder, self).__init__()
        self.in_size = in_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size

        self.featurizer = nn.Sequential(
            nn.Linear(in_size, 1024),
            nn.Linear(1024, seq_len * hidden_size),
            nn.LeakyReLU(),
        )
        self.final_featurizer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
    
    def forward(self, x):

        x = self.featurizer(x)
        # x [b, seq_len * hidden_size]
        x = x.view(-1, self.seq_len, self.hidden_size)
        x = self.final_featurizer(x)

        # returns batch, no_of_words, word_emb 
        return x
     
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
    
class ResidualBlock(nn.Module):
    """Represents 1D version of the residual block: https://arxiv.org/abs/1512.03385"""

    def __init__(self, input_dim):
        """Initializes the module."""
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LeakyReLU(),
            nn.Linear(input_dim, input_dim),
        )

    def forward(self, x):
        """Performs forward pass of the module."""
        skip_connection = x
        x = self.block(x)
        x = skip_connection + x
        return x
    
class LatentToMol(nn.Module):
    def __init__(self, 
                 in_size,
                 hidden_size, 
                 n_layers,
                 n_heads, 
                 seq_len, 
                 vocab, 
                 dropout=0.1):
        
        super(LatentToMol, self).__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(len(vocab), hidden_size, padding_idx=vocab.pad_index)
        self.pe = PositionalEncodings(d_model=hidden_size, p_dropout=dropout, seq_len=seq_len)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, hidden_size))
        self.classifier = nn.Linear(hidden_size, len(vocab))
        
        transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size,
                                                               nhead=n_heads,
                                                               dropout=dropout,
                                                               norm_first=True,
                                                               activation="gelu",
                                                               batch_first=True)
        
        self.trfmdecoder = nn.TransformerDecoder(decoder_layer=transformer_decoder_layer, 
                                                 num_layers=n_layers,
                                                 )
        self.latentencoder = LatentEncoder(in_size=in_size, 
                                       hidden_size=hidden_size,
                                       dropout=dropout)
        
        self.res_block = nn.Sequential(
            ResidualBlock(hidden_size),
            nn.LeakyReLU()
            )
        self.drop = nn.Dropout(0.1)

    def forward(self, latent, smi, tgt_mask=None, tgt_padding_mask=None):
        latent = self.latentencoder(latent)
        smi = self.embed(smi)
        # smi = self.res_block(smi)
        # smi = self.pe(smi)
        pe = self.pos_emb[:,:self.seq_len,:].to(latent.device)
        smi = self.drop(smi + pe)
        # smi = smi.permute
        x = self.trfmdecoder(
            tgt=smi,
            memory=latent,
            tgt_key_padding_mask = tgt_padding_mask,
            tgt_mask = tgt_mask
        )
        x = self.ln_f(x)
        out = self.classifier(x)
        # out = F.log_softmax(out, 2)
        return out