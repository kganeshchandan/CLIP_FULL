##################################################
# Imports
##################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam

# Custom
from models.transformer import MultiHeadAttention


# Utils
class Transpose(nn.Module):
    def __init__(self, d0, d1): 
        super(Transpose, self).__init__()
        self.d0, self.d1 = d0, d1

    def forward(self, x):
        return x.transpose(self.d0, self.d1)

_MODELS_CONFIG = {
    'vit-base': {'num_layers': 12, 'h_dim': 768, 'd_ff': 3072, 'num_heads': 12},
    'vit-large': {'num_layers': 24, 'h_dim': 1024, 'd_ff': 4096, 'num_heads': 16},
    'vit-huge': {'num_layers': 32, 'h_dim': 1280, 'd_ff': 5120, 'num_heads': 16},
}


##################################################
# ViT Transformer Encoder Layer
##################################################

class ViTransformerEncoderLayer(nn.Module):
    """
    An Image is Worth 16x16 Words: Transformers for Image Recognition At Scale, Dosovitskiy et al, 2020.
    https://arxiv.org/pdf/2010.11929.pdf
    """
    def __init__(self, h_dim, num_heads, d_ff=2048, dropout=0.0):
        super(ViTransformerEncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(h_dim)
        self.mha = MultiHeadAttention(h_dim, num_heads)
        self.norm2 = nn.LayerNorm(h_dim)
        self.ffn = nn.Sequential(
            nn.Linear(h_dim, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, h_dim)
        )

    def forward(self, x, mask=None):
        x_ = self.norm1(x)
        x = self.mha(x_, x_, x_, mask=mask) + x
        x_ = self.norm2(x)
        x = self.ffn(x_) + x
        return x


##################################################
# Vit Transformer Encoder
##################################################

class ViTransformerEncoder(nn.Module):
    """
    An Image is Worth 16x16 Words: Transformers for Image Recognition At Scale, Dosovitskiy et al, 2020.
    https://arxiv.org/pdf/2010.11929.pdf
    """
    def __init__(self, num_layers, h_dim, num_heads, d_ff=2048, 
                 max_time_steps=None, use_clf_token=False, dropout=0.0, dropout_emb=0.0):
        super(ViTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            ViTransformerEncoderLayer(h_dim, num_heads, d_ff=d_ff, dropout=dropout) 
            for _ in range(num_layers)
        ])
        self.pos_emb = nn.Embedding(max_time_steps, h_dim)
        self.use_clf_token = use_clf_token
        if self.use_clf_token:
            self.clf_token = nn.Parameter(torch.randn(1, h_dim))
        self.dropout_emb = nn.Dropout(dropout_emb)

    def forward(self, x, mask=None):
        if self.use_clf_token:
            clf_token = self.clf_token.unsqueeze(0).repeat(x.shape[0], 1, 1)
            x = torch.cat([clf_token, x], 1)
            if mask is not None:
                raise Exception('Error. clf_token with mask is not supported.')
        embs = self.pos_emb.weight[:x.shape[1]]
        x += embs
        x = self.dropout_emb(x)
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x


##################################################
# Visual Transformer (ViT)
##################################################

class ViT(nn.Module):
    """
    An Image is Worth 16x16 Words: Transformers for Image Recognition At Scale, Dosovitskiy et al, 2020.
    https://arxiv.org/pdf/2010.11929.pdf
    """
    def __init__(self, patch_size, num_layers, h_dim, num_heads, output_size, 
                 d_ff=2048, max_time_steps=None, use_clf_token=True, dropout=0.0, dropout_emb=0.0):
        super(ViT, self).__init__()
        self.proc = nn.Sequential(
            nn.Unfold((1, patch_size), 
                      stride=(1, patch_size)),
            Transpose(1, 2),
            nn.Linear(patch_size, h_dim),
        )
        self.enc = ViTransformerEncoder(num_layers, h_dim, num_heads, 
                                         d_ff=d_ff, 
                                         max_time_steps=max_time_steps, 
                                         use_clf_token=use_clf_token, dropout=dropout, dropout_emb=dropout_emb)
        self.mlp = nn.Linear(h_dim, output_size)

    def forward(self, x):
#         print("direct input", x.shape)
        x = self.proc(x)
#         print("after proc", x.shape)
        x = self.enc(x)
#         print("afetr enc", x.shape)
        x = x[:, 0] if self.enc.use_clf_token else x.mean(1)
        x = self.mlp(x)
#         print("out", x.shape)
        return x


# Get visual transformer
def get_vit(args):
    model_args = {
        'patch_size': args.patch_size, 
        'num_layers': _MODELS_CONFIG[args.model]['num_layers'],
        'h_dim': _MODELS_CONFIG[args.model]['h_dim'],
        'num_heads': _MODELS_CONFIG[args.model]['num_heads'], 
        'num_classes': args.num_classes, 
        'd_ff': _MODELS_CONFIG[args.model]['d_ff'], 
        'max_time_steps': args.max_time_steps, 
        'use_clf_token': args.use_clf_token,
        'dropout': args.dropout,
        'dropout_emb': args.dropout_emb,
    }
    model = ViT(**model_args)
    if len(args.model_checkpoint) > 0:
        model = model.load_from_checkpoint(args.model_checkpoint, **model_args)
        print(f'Model checkpoint loaded from {args.model_checkpoint}')
    return model


##################################################
# Classifier - PyTorchLightning Wrapper for ViT
##################################################

class Classifier(pl.LightningModule):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.args = args
        self.model = get_vit(args)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx, part='train'):
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        acc = (1.0 * (F.softmax(logits, 1).argmax(1) == y)).mean()

        self.log(f'{part}_loss', loss, prog_bar=True)
        self.log(f'{part}_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx, part='val')

    def configure_optimizers(self):
        return Adam(self.parameters(), self.args.lr, weight_decay=0.005)
