from esm_train import get_esm
from training import create_loss, run_training, parser
from classifiers import RecyclingClassifier, RecyclingClassifierConfig

def create_model(args):
    esm, tokenizer = get_esm(args.type)

    config = RecyclingClassifierConfig(n_labels=1,loss=create_loss(args), base_type=args.type, n_recycle_steps=args.n_recycle_steps,
                                       n_heads = args.n_heads, n_enc_layers = args.n_enc_layers, dropout_rate=args.dropout,
                                       dim_ffw=args.dim_ffw)
    model = RecyclingClassifier(base_model=esm, config=config)
    model.set_base_requires_grad(False)
    return model, tokenizer

def add_arguments(parser):
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_recycle_steps', type=int, default=3)
    parser.add_argument('--n_enc_layers', type=int, default=3)
    parser.add_argument('--dim_ffw', type=int, default=512)

def main(args):
    run_training(args, create_model)

if __name__ == '__main__':
    add_arguments(parser)
    args = parser.parse_args()
    main(args)
