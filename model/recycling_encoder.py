import torch

from utils import get_esm
from training import create_loss, run_training, parser
from dataclasses import dataclass
from token_classifier_base import TokenClassifierConfig, TokenClassifier
from modules import RecyclingEncoder


@dataclass
class RecyclingClassifierConfig(TokenClassifierConfig):
    n_recycle_steps : int = 3
    dim_model : int|None = None
    dim_ffw : int = 2048
    n_heads : int = 8
    n_enc_layers : int = 3
    kernel_size = 31
    use_cnn : bool = False
    # Use a gated SwiGLU feedforward block in the encoder layers instead of the dense one
    swiglu : bool = False

    
class RecyclingClassifier(TokenClassifier):
    def __init__(self, base_model, config):
        super().__init__(config, base_model)
        
        # Pos embeds from ESM-2 already in the embeddings
        # self.pos_embed = SinPositionalEncoding(config.dim_model, 1024) # ESM has max 1024 tokens, incl. [cls]...[eos]
        model_dim = base_model.config.hidden_size if not config.dim_model else config.dim_model
        self.create_projection_layer(config)
        self.encoder = RecyclingEncoder(model_dim, config.n_heads, config.n_enc_layers, config.n_recycle_steps,
                                        dropout=config.dropout_rate, d_feedforward=config.dim_ffw,
                                        swiglu=config.swiglu)
        self.output = torch.nn.Linear(model_dim, config.n_labels)

    def create_projection_layer(self, config):
        if config.dim_model:
            input_dim = self.base.config.hidden_size
            if config.use_cnn:
                self.project = torch.nn.Conv1d(in_channels=input_dim, out_channels=config.dim_model, kernel_size=config.kernel_size, padding=config.kernel_size // 2)
            else:
                self.project = torch.nn.Linear(input_dim, config.dim_model)
        else:
            self.project = torch.nn.Identity()

    def forward(self, input_ids, attention_mask, **kwargs):
        base_out = self.base(input_ids=input_ids, attention_mask=attention_mask)
        x = base_out[0]

        if self.config.use_cnn:
            x = x.transpose(1, 2)
        x = self.project(x)

        if self.config.use_cnn:
            x = x.transpose(1, 2)

        # x = x + self.pos_embed(x)
        if 'no_flash_attn' in kwargs and kwargs['no_flash_attn']:
            # Transform the inputs to sequence-first. Expecting batch size of 1
            x = x.moveaxis(0, 1).squeeze()
            x = self.encoder(x)
            x = x.unsqueeze(0)
        else:
            x = self.encoder(x, mask=attention_mask)

        return self.output(x), base_out

def create_model(args):
    esm, tokenizer = get_esm(args.type)

    config = RecyclingClassifierConfig(n_labels=1,loss=create_loss(args), base_type=args.type, n_recycle_steps=args.n_recycle_steps,
                                       n_heads = args.n_heads, n_enc_layers = args.n_enc_layers, dropout_rate=args.dropout,
                                       dim_ffw=args.dim_ffw, use_cnn=args.use_cnn, dim_model=args.dim_model,
                                       swiglu=args.swiglu)
    model = RecyclingClassifier(base_model=esm, config=config)
    model.set_base_requires_grad(False)
    return model, tokenizer

def add_arguments(parser):
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_recycle_steps', type=int, default=3)
    parser.add_argument('--n_enc_layers', type=int, default=3)
    parser.add_argument('--dim_ffw', type=int, default=512)
    parser.add_argument('--dim_model', type=int, default= None)
    parser.add_argument('--use_cnn', type=bool, default=False)
    parser.add_argument('--swiglu', action='store_true', default=False,
                        help='Use a gated SwiGLU feedforward block in the encoder layers instead of the dense one.')

def main(args):
    run_training(args, create_model)

if __name__ == '__main__':
    add_arguments(parser)
    args = parser.parse_args()
    main(args)
