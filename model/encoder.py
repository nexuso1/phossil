# This is the main entrypoint of training encoder-based models

import numpy as np
import torch

from torch import Module
from modules import Conv1dModel, ConvLayerConfig, SinPositionalEncoding, FusedMBConv1dModel, ResidualMLP, FusedMBConvConfig
from token_classifier_base import TokenClassifier, TokenClassifierConfig
from argparse import ArgumentParser
from training import run_training, parser, create_loss
from utils import get_esm
from dataclasses import dataclass, field



@dataclass
class EncoderClassifierConfig(TokenClassifierConfig):
    hidden_size : int = 256
    encoder_dim : int = 256
    n_heads : int = 8
    n_layers : int = 1
    sr_dim : int = 256
    sr_n_tokens : int = 1
    pos_embed_type : str = 'sin'
    sr_type : str = 'cnn'
    cnn_type : str = 'basic'
    sr_type : str = 'cnn'
    ffw_dim : int = 2048
    sr_cnn_layers : list[ConvLayerConfig|FusedMBConvConfig] = field(default_factory= lambda :[
            ConvLayerConfig(1280, 256, 5, 2, 2),
            ConvLayerConfig(256, 378, 5, 2, 2),
            ConvLayerConfig(378, 512, 5, 2, 2),
            ConvLayerConfig(512, 1024, 5, 2, 2),
        ])
    
    res_cnn_layers : list[ConvLayerConfig] = field(default_factory= lambda :[
            ConvLayerConfig(1280, 256, 31, 1, 1),
        ])
    
    mlp_layers : list[int] = field(default_factory=lambda : [256, 256, 256])


class EncoderClassifier(TokenClassifier):
    def __init__(self, config: EncoderClassifierConfig, base_model: Module) -> None:
        super().__init__(config, base_model)
        enc_layer = torch.nn.TransformerEncoderLayer(config.encoder_dim, nhead=config.n_heads,
                                                    dim_feedforward=config.ffw_dim,
                                                    activation='relu', batch_first=True)
        
        # Setup positional embeddings
        if config.pos_embed_type == 'sin':
            self.pos_embed = SinPositionalEncoding(config.encoder_dim, 1024 + config.sr_n_tokens) # [seq_rep][cls]...[eos]

        else:
            self.pos_embed = None

        self.encoder = torch.nn.TransformerEncoder(enc_layer, norm=torch.nn.LayerNorm(config.encoder_dim), num_layers=config.n_layers)

        if config.sr_type == 'cnn':
            self.create_sr_cnn() # Creates and initializes the seq. rep CNN
        
        # Create the residue representation CNN
        if config.cnn_type == 'basic':
            self.res_cnn = Conv1dModel(config.res_cnn_layers, pool=False, dropout=0)
        elif config.cnn_type == 'fused':
            self.res_cnn = FusedMBConv1dModel(config.res_cnn_layers, pool=False, dropout=0)

        # Create a residual MLP classifier
        self.classifier_mlp = ResidualMLP(self.config.mlp_layers,
                                       input_size=config.encoder_dim, activation=torch.nn.ReLU(), norm=torch.nn.LayerNorm,
                                       dropout=config.dropout_rate)
        
        self.classifier = torch.nn.Sequential(self.classifier_mlp, torch.nn.Linear(self.config.mlp_layers[-1], self.config.n_labels))
        # Initialize the modules
        init_list = [self.encoder, self.classifier, self.res_cnn]
        for module in init_list:
            module.apply(self.xavier_init)
        
        # Print info about this model
        print(self)
    
    def create_sr_cnn(self):
        # Create sequence-representation CNN
        if self.config.cnn_type == 'basic':
            self.sr_cnn = Conv1dModel(self.config.sr_cnn_layers, dropout=0, pool=True, activ=torch.nn.ReLU())
        elif self.config.cnn_type == 'fused':
            self.sr_cnn = FusedMBConv1dModel(self.config.sr_cnn_layers, pool=True, dropout=0, activ=torch.nn.ReLU())
        
        self.seq_rep = self.sr_cnn
        # Initialize weights
        self.seq_rep.apply(self.xavier_init)
        
    def get_mean_sequence_reps(self, sequence_output : torch.Tensor, batch_lens):
        # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
        pad_mask = torch.arange(0, sequence_output.shape[0], device=self.device)[:, None, None].expand_as(sequence_output)
        lens_reshaped = batch_lens[:, None, None].expand_as(pad_mask)
        # Prepare the mask
        pad_mask = pad_mask > lens_reshaped # True if a given position is padding
        # Zero the padding values
        sequence_output[pad_mask] = 0
        # Zero the BOS token
        sequence_output[:, 0, :] = 0 
        # Calculate the sequence means
        seq_rep = torch.mean(sequence_output, 1)

        return seq_rep

    def forward(self, input_ids, attention_mask, **kwargs):
        base_out = self.base(input_ids=input_ids, attention_mask=attention_mask)
        x = base_out[0]
        proj = self.res_cnn(x)
        
        if self.config.sr_type == 'mean':
            seq_rep = self.get_mean_sequence_reps(x, kwargs['batch_lens'])
        else:
            seq_rep = self.seq_rep(x)
        
        enc_mask = torch.cat([torch.ones(attention_mask.shape[0], 1, device=self.device), attention_mask], 1)
        if len(proj.shape) < 3:
            proj = proj.unsqueeze(0)
            seq_rep = seq_rep.unsqueeze(0)
        x = torch.cat([seq_rep.unsqueeze(1) , proj], axis=1)
        x = x + self.pos_embed(x)
        if 'no_flash_attn' in kwargs and kwargs['no_flash_attn']:
            # Transform the inputs to sequence-first. Expecting batch size of 1
            x = x.moveaxis(0, 1).squeeze()
            x = self.encoder(x)
            x = x.unsqueeze(0)
        else:
            x = self.encoder(x,src_key_padding_mask=torch.bitwise_not(enc_mask.bool()))
        return self.classifier(x)[:, 1:], base_out


def setup_config(args, config, base_config):
    mlp_layers = [args.hidden_size for _ in range(args.n_layers_mlp)]
    sr_sizes = 2 ** np.linspace(np.log2(args.sr_init_size), np.log2(args.sr_final_size), args.sr_n)
    base_size = base_config.hidden_size

    if args.cnn_type == 'basic':
        # First layer
        sr_cnn_layers = [ConvLayerConfig(base_size, int(sr_sizes[0]), args.sr_kernel_size, args.block_size, 2)]
        # Rest of layers
        sr_cnn_layers = sr_cnn_layers + \
            [ConvLayerConfig(int(sr_sizes[i]), int(sr_sizes[i+1]), args.sr_kernel_size, args.block_size, 2)
            for i in range(args.sr_n - 1)]
        
        # Should only have one layer
        res_cnn_layers = [ConvLayerConfig(base_size, args.encoder_dim, args.res_kernel_size, 1, 1)]

    elif args.cnn_type == 'fused':
        sr_cnn_layers = [FusedMBConvConfig(base_size, int(sr_sizes[0]), args.sr_kernel_size, args.block_size, 2, args.expand)]

        sr_cnn_layers = sr_cnn_layers + \
            [FusedMBConvConfig(int(sr_sizes[i]), int(sr_sizes[i+1]), args.sr_kernel_size, args.block_size, 2, args.expand) 
            for i in range(args.sr_n - 1)]
        res_cnn_layers = [FusedMBConvConfig(base_size, args.encoder_dim, args.res_kernel_size, 1, 1, args.expand)]
    
    config.mlp_layers = mlp_layers
    config.sr_cnn_layers = sr_cnn_layers
    config.res_cnn_layers = res_cnn_layers
    config.n_layers = args.n_layers
    config.dropout_rate = args.dropout
    config.sr_dim = args.encoder_dim
    config.sr_type = args.sr_type
    config.cnn_type = args.cnn_type
    config.encoder_dim = args.encoder_dim
    config.ffw_dim = args.ffw_dim
    
    return config

def create_model(args):
    base, tokenizer = get_esm(args.type)
    conf = EncoderClassifierConfig(1, loss = create_loss(args), mlp_layers=[], sr_cnn_layers=[], res_cnn_layers=[])
    setup_config(args, conf, base.config)
    
    classifier = EncoderClassifier(conf, base)
 
    if not args.lora:
        classifier.set_base_requires_grad(False)

    return classifier, tokenizer

def add_arguments(parser : ArgumentParser):
    parser.add_argument('--n_layers_mlp', type=int, help='Number of MLP classifier layers', default=3)
    parser.add_argument('--block_size', type=int, help='Number of seq. rep. CNN layers in one block', default=1)
    parser.add_argument('--cnn_type', type=str, help=
                        '''Type of cnn to be used for seq/residue reps. Options are "basic",
                        for standard Conv1d layers, or "fused", for FusedMBConv layers''',
                        default='basic')
    parser.add_argument('--expand_m', type=int, help='If using FusedMBConv, this is the expansion multiplier.', default=4)
    parser.add_argument('--sr_n', type=int, help='Number of seq. rep. CNN blocks', default=3)
    parser.add_argument('--sr_kernel_size', type=int, help='Seq. rep. kernel size', default=5)
    parser.add_argument('--sr_init_size', type=int, help='Initial dimension for the seq. rep. CNN', default=256)
    parser.add_argument('--sr_final_size', type=int, help='Final dimension for the seq. rep. CNN', default=256)
    parser.add_argument('--res_kernel_size', type=int, help='Residue representation kernel size', default=31)
    parser.add_argument('--sr_type', type=str, help='Sequence representation type. Either "mean" or "cnn".', default='cnn')
    parser.add_argument('--encoder_dim', type=int, help='Encoder model dimension', default=256)
    parser.add_argument('--ffw_dim', type=int, help='Encoder FFW dimension', default=2048)
    parser.add_argument('--expand', type=int, help='Expansion constant for fused mbconv', default=4)

def main(args):
    run_training(args, create_model)

if __name__ == '__main__':
    add_arguments(parser)
    args = parser.parse_args()
    main(args)