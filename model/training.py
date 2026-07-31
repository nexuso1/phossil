# This source file is the backbone of model training. When training
# a model, we use the run_training function defined in this 
# file

import torch
import lightning as L
import os.path
import datetime
import matplotlib.pyplot as plt
import io
import json
import pandas as pd

from torchmetrics import F1Score, MatthewsCorrCoef, Precision, Recall, AUROC, \
MeanMetric, AveragePrecision, PrecisionRecallCurve, MetricCollection, ConfusionMatrix
from torch.utils.data import DataLoader
from data_loading import prep_batch
from functools import partial
from torchvision.transforms import ToTensor
from PIL import Image
from token_classifier_base import TokenClassifier
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, Callback
from utils import Metadata, sigmoid_focal_loss
from lightning.pytorch.loggers import TensorBoardLogger
from data_loading import prepare_datasets, prepare_full_dataset
from transformers import AutoTokenizer
from pathlib import Path
from argparse import Namespace, ArgumentParser

parser = ArgumentParser()

parser.add_argument('--seed', type=int, help='Random seed', default=42)
parser.add_argument('--batch_size', type=int, help='Batch size)', default=4)
parser.add_argument('--epochs', type=int, help='Number of training epochs', default=20)
parser.add_argument('--frozen_epochs', type=int, default=0, help='Number of training epochs with the base model frozen')
parser.add_argument('--prot_info_path', type=str, 
                     help='Path to the protein dataset. Expects a dataframe with columns ("id", "sequence", "sites"). "sequence" is the protein AA string, "sites" is a list of phosphorylation sites.',
                     default='../data/dbptm/dbptm_info.json')
parser.add_argument('--dataset_path', type=str, help='Path to the prepared dataset, with information about train, test sets; and folds.', default='../data/dbptm/splits_S.json')
parser.add_argument('--weight_decay', type=float, help='Weight decay', default=1e-4)
parser.add_argument('--accum', type=int, help='Number of gradient accumulation steps', default=3)
parser.add_argument('--hidden_size', type=int, help='Classifier hidden size', default=128)
parser.add_argument('--lr', type=float, help='Learning rate', default=3e-4)
parser.add_argument('--frozen_lr', type=float, help='Frozen phase starting learning rate', default=1e-3)
parser.add_argument('-o', type=str, help='Output folder', default=None)
parser.add_argument('-n', type=str, help='Model name', default='esm')
parser.add_argument('--compile', action='store_true', default=False, help='Compile the model')
parser.add_argument('--lora', action='store_true', help='Use LoRA', default=False)
parser.add_argument('--dropout', type=float, help='Dropout probability', default=0)
parser.add_argument('--type', help='ESM Model type', type=str, default='650M')
parser.add_argument('--pos_weight', help='Positive class weight', type=float, default=3)
parser.add_argument('--num_workers', help='Number of multiprocessing workers', type=int, default=0)
parser.add_argument('--n_layers', help='Number of RNN/Transformer classifier layers', type=int, default=1)
parser.add_argument('--checkpoint_path', help='Resume training from checkpoint', type=str, default=None)
parser.add_argument('--model_path', help='Load model from this path (not a checkpoint)', type=str, default=None)
parser.add_argument('--focal', help='Use focal loss. In this mode, pos_weight will be treated as the alpha parameter.', action='store_true', default=False)
parser.add_argument('--residues', help='List of residues to train on', default="['S', 'T', 'Y']", type=str)
parser.add_argument('--ignore_label', help='Label that will be ignored by the loss', default=-1, type=int)
parser.add_argument('--patience', help='Patience during training', default=5, type=int)
parser.add_argument('--debug', help='Debug mode', default=False, action='store_true')
parser.add_argument('--step_lr', help='Use StepLR scheduler', default=False, action='store_true')
parser.add_argument('--release', help='Train in the release mode (using ALL data available, no test set)', default=False, action='store_true')
parser.add_argument('--modify_prob', help='Probabilty of modifying input residues during training', default=0, type=float)
parser.add_argument('--mask_prob', help='Relative probabilty of masking input residues during training. Only relevant if "modify_prob" > 0', default=0.7, type=float)
parser.add_argument('--rand_prob', help='Relative probability of randomly changing input residues during training. Only relevant if "modify_prob" > 0',
                     default=0.15, type=float)
parser.add_argument('--sub_prob', help='Relative probability of substituting input residues via a substitution matrix (BLOSUM62) during training. Only relevant if "modify_prob" > 0',
                     default=0.15, type=float)
parser.add_argument('--fix_decay', action='store_true', help="Do not apply weight decay to bias and layer norm parameters.")
parser.add_argument('--kinase', action='store_true', help='Use kinase labels in prediction')
parser.add_argument('--unfreeze_indices', type=str, default="[]", help="Indices of base model layers to unfreeze")
parser.add_argument('--fold', type=int, default=None, help='Train only this fold index. Leave None for default all-fold training')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class LightningWrapper(L.LightningModule):
    def __init__(self, args, module : TokenClassifier, epoch_metrics : MetricCollection,
                 step_metrics : MetricCollection, ds_size : int, logdir : str, train_epochs, lr):
        super(LightningWrapper, self).__init__()
        self.classifier = module
        self.ds_size = ds_size
        self.train_epochs = train_epochs
        self.logdir = logdir
        self.predicting_kinases = args.kinase
        self.lr = lr

        for prefix in ['train_', 'val_', 'test_']:
            for kind, metrics in zip(['epoch', 'step'], [epoch_metrics, step_metrics]):
                self.__setattr__(f'{prefix}{kind}_metrics', metrics.clone(prefix=prefix))

                # Add kinase metrics if we are using them
                if self.predicting_kinases:
                    self.__setattr__(f'{prefix}kinase_{kind}_metrics', metrics.clone(prefix=f'{prefix}kinase_'))
        
        self.loss_metric = MeanMetric()
        self.test_preds = []
        # self.optimal_threshold = 0
        # self.optimal_dev_metric_values = {
        #     'f1' : 0,
        #     'precision' : 0,
        #     'recall' : 0
        # }
        self.debug = False
        if hasattr(args, "debug") and args.debug:
            self.debug = True
            self.print_counter = 0
            
        self.save_hyperparameters(args)

    def _compute_metrics_step(self, logits, labels, step_metrics, epoch_metrics, batch_size=None, **kwargs):
        step_vals = step_metrics(logits, labels)
        epoch_metrics.update(logits, labels.int())
        self.log_dict(step_vals, sync_dist=True, prog_bar=True, logger=True, batch_size=batch_size)

    def training_step(self, batch, batch_idx):
        loss, logits, outputs = self.classifier.train_predict(**batch)
        self.log('train_loss', loss, logger=True, prog_bar=True, sync_dist=True, batch_size=self.hparams.batch_size)
        self._compute_metrics_step(logits.view(-1), batch['labels'].view(-1),
                                   self.train_step_metrics, self.train_epoch_metrics)
        
        if self.predicting_kinases:
            self._compute_metrics_step(outputs['kinase_logits'][batch['labels'] == 1].view(-1), batch['kinase_labels'].view(-1),
                                       self.train_kinase_step_metrics, self.train_kinase_epoch_metrics, batch_size=int((batch['labels'] == 1).sum()))
        
        if self.debug:
            if self.print_counter >= 250:
                with torch.no_grad():
                    print(batch)
                    print(torch.sigmoid(logits))
                self.print_counter = 0
            
            self.print_counter += 1

        return loss

    def _eval_step(self, batch, batch_idx, type : str):
        loss, logits, outputs = self.classifier.predict(**batch)
        self.log(f'{type}_loss', loss, logger=True, prog_bar=True, sync_dist=True, batch_size=self.hparams.batch_size)
        self._compute_metrics_step(logits.view(-1), batch['labels'].view(-1), 
                                   self.__getattr__(f'{type}_step_metrics'), self.__getattr__(f'{type}_epoch_metrics'))
        
        if self.predicting_kinases:
            self._compute_metrics_step(outputs['kinase_logits'][batch['labels'] == 1].view(-1), batch['kinase_labels'].view(-1),
                                       self.__getattr__(f'{type}_kinase_step_metrics'), self.__getattr__(f'{type}_kinase_epoch_metrics'),
                                       batch_size=int((batch['labels'] == 1).sum()))
            
        return loss, logits, outputs
    
    def validation_step(self, batch, batch_idx):
        self._eval_step(batch, batch_idx, 'val')
    
    def test_step(self, batch, batch_idx):
        loss, logits, outputs = self._eval_step(batch, batch_idx, 'test')

        #self.optimum_metrics.update(logits.view(-1), batch['labels'].view(-1))
        # if self.predicting_kinases:
        #     self.optimum_kinase_metrics.update(outputs['kinase_logits'][batch['labels'] == 1].view(-1), batch['kinase_labels'].view(-1))

        # Relevant positions mask
        mask = batch['labels'] != -1
        # Save test predictions
        for i in range(mask.shape[0]):
            preds = (logits[i][mask[i]].cpu().numpy().reshape(-1), # prediction logits
                                    batch['labels'][i][mask[i]].cpu().numpy().reshape(-1), # labels
                                    (torch.nonzero(mask[i] ) - 1).cpu().numpy().reshape(-1), # sequence indices (0-based) of relevant positions
                                    int(batch['indices'][i].cpu().numpy()))
            if self.predicting_kinases:
                preds = preds + (outputs['kinase_logits'][i][mask[i]].cpu().numpy(),)

            self.test_preds.append(preds) # index into the test set

    def _shared_epoch_start(self, type):

        # Make sure all metrics are reset
        for m_type in ['step','epoch']:
            self.__getattr__(f'{type}_{m_type}_metrics').reset()

            if self.predicting_kinases:
                self.__getattr__(f'{type}_kinase_{m_type}_metrics').reset()

    def on_validation_epoch_start(self):
        self._shared_epoch_start('val')

    def on_test_epoch_start(self):
        # self.optimum_metrics = MetricCollection({
        #         'optimum_f1' : F1Score('binary', threshold=self.optimal_threshold, ignore_index=self.classifier.ignore_index),
        #         'optimum_recall' : Recall('binary', threshold=self.optimal_threshold, ignore_index=self.classifier.ignore_index),
        #         'optimum_precision' : Precision('binary', threshold=self.optimal_threshold, ignore_index=self.classifier.ignore_index),
        #         'optimum_mcc' : MatthewsCorrCoef('binary', threshold=self.optimal_threshold, ignore_index=self.classifier.ignore_index),
        #         'optimum_confusion_matrix' : ConfusionMatrix('binary', threshold=self.optimal_threshold, ignore_index=self.classifier.ignore_index),
        #     }
        # ).to(device=self.device)
    
        # if self.predicting_kinases:
        #     self.optimum_kinase_metrics = self.optimum_metrics.clone('kinase_')
        
        self._shared_epoch_start('test')

    def on_train_epoch_start(self):
        self._shared_epoch_start('train')

    def _log_epoch_metrics(self, epoch_metrics, save_image=True):
        epoch_metrics.compute()
        log_directly = { k : v for k, v in epoch_metrics.items() if not (k.endswith('prcurve') or k.endswith('confusion_matrix'))}
        self.log_dict(log_directly, prog_bar=True, logger=True, sync_dist=True)
        
        if not save_image:
            return
        
        log_image_metrics = { k for k in epoch_metrics.keys() if (k.endswith('prcurve') or k.endswith('confusion_matrix'))}
        for metric in log_image_metrics:
            fig, ax = plt.subplots(figsize=(10, 10))
            try:
                if metric.endswith('prcurve'):
                    epoch_metrics[metric].plot(ax=ax, score=True)
                else:
                    epoch_metrics[metric].plot(ax=ax)
            except IndexError:
                print(epoch_metrics[metric].compute())
                break
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            im = ToTensor()(Image.open(buf))

            self.logger.experiment.add_image(
                metric,
                im,
                global_step=self.current_epoch,
            )
            plt.close(fig=fig)


    def find_optimal_threhsold(self):
        precision, recall, thresholds = self.val_epoch_metrics['val_prcurve'].compute()
        # Slice precision and recall to match the exact size of the thresholds tensor
        precision_bound = precision[:len(thresholds)]
        recall_bound = recall[:len(thresholds)]

        f1_scores = (2 * precision_bound * recall_bound) / (precision_bound + recall_bound + 1e-10)
        best_idx = torch.argmax(f1_scores)
        if f1_scores[best_idx].cpu().numpy() < self.optimal_dev_metric_values['f1']:
            return
        
        optimal_threshold = thresholds[best_idx]
        print(f'optimum_changed: {optimal_threshold}')
        self.optimal_threshold = float(optimal_threshold.cpu())
        
        self.optimal_dev_metric_values = {
            'f1' : float(torch.max(f1_scores).cpu().numpy()),
            'precision' : float(precision_bound[best_idx].cpu()),
            'recall' : float(recall_bound[best_idx].cpu())
        }

        if self.predicting_kinases:
            precision, recall, thresholds = self.val_kinase_epoch_metrics['val_kinase_prcurve'].compute()
            precision_bound = precision[:len(thresholds)]
            recall_bound = recall[:len(thresholds)]

            f1_scores = (2 * precision_bound * recall_bound) / (precision_bound + recall_bound + 1e-10)
            best_idx = torch.argmax(f1_scores)

            self.optimal_kinase_threshold = thresholds[best_idx]
            self.optimal_kinase_dev_metric_values = {
                'f1' : float(torch.max(f1_scores).cpu().numpy()),
                'precision' : float(precision_bound[best_idx].cpu()),
                'recall' : float(recall_bound[best_idx].cpu())
            }
    
    def _shared_epoch_end(self, mode, save_image=True):
        epoch_metrics = self.__getattr__(f'{mode}_epoch_metrics')
        self._log_epoch_metrics(epoch_metrics, save_image=save_image)

    def on_train_epoch_end(self) -> None:
        self._shared_epoch_end('train', save_image=False)
        if self.predicting_kinases:
            self._shared_epoch_end('train_kinase', save_image=False)

    def on_validation_epoch_end(self) -> None:
        # self.find_optimal_threhsold()
        self._shared_epoch_end('val')
        if self.predicting_kinases:
            self._shared_epoch_end('val_kinase')
        
    def on_test_epoch_end(self):
        self._shared_epoch_end('test')
        #self._log_epoch_metrics(self.optimum_metrics)
        if self.predicting_kinases:
            self._shared_epoch_end('test_kinase')
            #self._log_epoch_metrics(self.optimum_kinase_metrics)

    def get_parameter_names(self, model : torch.nn.Module, forbidden_layer_types):
        """
        Returns the names of the model parameters that are not inside a forbidden layer.
        
        Based on https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_pt_utils.py#L1026
        """
        result = []
        
        for name, child in model.named_children():
            child_params = self.get_parameter_names(child, forbidden_layer_types)
            result += [
                f"{name}.{n}"
                for n in child_params
                if not isinstance(child, tuple(forbidden_layer_types))
            ]
        result += [k for k in model._parameters]

        return result

    def configure_optimizers(self):
        optimizer_kwargs = {
                "betas": (0.9, 0.98),
                "eps": 1e-8,
                'lr' : self.lr,
            }
        if self.hparams.fix_decay:
            decay_parameters = self.get_parameter_names(self.classifier, [torch.nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_parameters = [
                {
                    "params": [p for n, p in self.classifier.named_parameters() if n in decay_parameters],
                    "weight_decay": self.hparams.weight_decay,
                },
                {
                    "params": [p for n, p in self.classifier.named_parameters() if n not in decay_parameters],
                    "weight_decay": 0.0,
                },
            ]
        else:
            optimizer_parameters = self.classifier.parameters()
            optimizer_kwargs['weight_decay'] = self.hparams.weight_decay

        optim = torch.optim.AdamW(optimizer_parameters, **optimizer_kwargs)
        if self.hparams.step_lr:
            # Needed for UniPTM training
            schedule = torch.optim.lr_scheduler.StepLR(optim, step_size=20, gamma=0.92)
        else:
            schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=self.train_epochs)
        return {'optimizer' : optim, 'lr_scheduler' : { 
            "scheduler" : schedule,
            "interval": "epoch",
            "monitor" : "train_loss",
            "frequency" : 1
        }}

def save_model(args, model : TokenClassifier, name : str):
    """
    Saves the model to the folder args.o if given, otherwise to args.logdir, with the batch['kinase_labels']given name.
    """
    if args.o is None:
        folder = args.logdir
    else:
        folder = args.o

    save_path = f'{folder}/{name}.pt'
    if not os.path.exists(f'{folder}'):
        os.mkdir(f'{folder}')

    model.save(save_path)
    print(f'Model saved to {save_path}')

def create_loss(args):
    # Create a loss function
    if args.focal:
        return partial(sigmoid_focal_loss, alpha=args.pos_weight, reduction='mean')
    
    return torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([args.pos_weight]))
    
def create_callbacks(logdir, patience, suffix=''):

    # Best model checkpoint
    best_callback = ModelCheckpoint(logdir, filename=f'best{suffix}', monitor='val_mcc', mode='max',
                                      save_on_train_epoch_end=1, auto_insert_metric_name=True)
    # Training checkpoint (because having a defined ModelCheckpoint overrides the default checkpointing)
    chkpt_callback = ModelCheckpoint(logdir, filename=f'chkpt{suffix}')
    es_callback = EarlyStopping('val_mcc', patience=patience, mode="max")
    return [best_callback, chkpt_callback, es_callback]

def train_model(args, train, dev, test, model : TokenClassifier, logdir, fold, metadata : Metadata,
                master_logdir):
    step_metrics, epoch_metrics = create_metrics(args.ignore_label)

    logger = TensorBoardLogger(logdir, name=f'tb_log')

    # Use deepspeed 
    if torch.cuda.device_count() > 1:
        strategy = "deepspeed_stage_2"
    else:
        strategy = "auto"
 

    # Frozen training
    if args.frozen_epochs > 0 and not metadata.data['frozen_finished'][fold]:
        callbacks = create_callbacks(logdir, patience=args.patience, suffix='_frozen')
        print('Frozen phase training')
        model.freeze_phase()
        if not isinstance(model, LightningWrapper):
            training_model = LightningWrapper(args, model, step_metrics=step_metrics, epoch_metrics=epoch_metrics, ds_size=len(train), logdir=logdir,
                                    train_epochs=args.frozen_epochs, lr=args.frozen_lr)
        trainer = L.Trainer(logger=logger, callbacks=callbacks, max_epochs=args.frozen_epochs,
                            deterministic=True, log_every_n_steps=1,  accumulate_grad_batches=args.accum, strategy=strategy,
                            default_root_dir=logdir, num_sanity_val_steps=0)

        trainer.fit(training_model, train, dev, ckpt_path=args.checkpoint_path)
        best = torch.load(f'{logdir}/best_frozen.ckpt')
        training_model.load_state_dict(best['state_dict'])
        model = training_model.classifier
        metadata.data['frozen_finished'][fold] = True
        metadata.save(master_logdir)

    print('Unfrozen phase training')
    callbacks = create_callbacks(logdir, patience=args.patience)
    # Unfrozen training
    trainer = L.Trainer(logger=logger, callbacks=callbacks, max_epochs=args.epochs,
                        deterministic=True, log_every_n_steps=1,  accumulate_grad_batches=args.accum, strategy=strategy,
                        default_root_dir=logdir, num_sanity_val_steps=0)
    model.unfreeze_phase()
    if not isinstance(model, LightningWrapper):
        training_model = LightningWrapper(args, model, step_metrics=step_metrics, epoch_metrics=epoch_metrics, ds_size=len(train), logdir=logdir,
                                train_epochs=args.epochs, lr=args.lr)
    trainer.fit(training_model, train, dev, ckpt_path=args.checkpoint_path)
    best = torch.load(f'{logdir}/best.ckpt')
    training_model.load_state_dict(best['state_dict'])
    test_metrics = trainer.test(training_model, test)

    # Save predictions into a DataFrame
    columns = ['logits', 'labels', 'sequence_indices', 'df_index']
    if args.kinase:
        columns += ['kinase_logits']

    pred_df = pd.DataFrame.from_records(training_model.test_preds, columns=columns)
    pred_df.to_json(f"{logdir}/test_preds_fold_{fold}.json")
    print(test_metrics)
    # print(f'Optimal prediction threshold (from validation data): {training_model.optimal_threshold}')
    
    # test_metrics[0]['optimal_dev_pred_threshold'] = training_model.optimal_threshold
    # for k, metric in training_model.optimal_dev_metric_values.items():
    #     test_metrics[0][f'dev_{k}'] = metric

    # if args.kinase:
    #     print(f'Optimal kinase prediction threshold (from validation data): {training_model.optimal_kinase_threshold}')
    #     for k, metric in training_model.optimal_kinase_dev_metric_values.items():
    #         test_metrics[0][f'dev_kinase_{k}'] = metric
    #     test_metrics[0]['optimal_dev_kinase_pred_threshold'] = training_model.optimal_threshold

    return model, test_metrics

def get_tokenizer(args):
    if args.type == '3B' :
        tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t36_3B_UR50D')
    elif type == '15B':
        tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t48_15B_UR50D')
    elif type == '35M':
        tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t12_35M_UR50D')
    else:
        tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D')

    return tokenizer

def prepare_model(args, create_model_fn):
    model, tokenizer = create_model_fn(args)

    return model, tokenizer

def handle_metadata(args, n_folds=5):
    if not args.checkpoint_path:
        # Create metadata

        meta = Metadata()
        meta.data = {'args' : args }
        meta.data['current_fold'] = 0
        meta.data['test_metrics'] = [{} for _ in range(n_folds)]
        meta.data['frozen_finished'] = [False for _ in range(n_folds)]
        meta.save(args.logdir)
    else:
        # Parse info from the metadata file

        par_dir = Path(args.checkpoint_path).parent
        chkpt_path = args.checkpoint_path
        with open(f'{par_dir.parent}/metadata.json', 'r') as f:
            meta = Metadata(**json.load(f))
            
            # Fix old way of saving metada if applicable
            if 'current_fold' not in meta.data:
                meta.data['current_fold'] = int(par_dir.name[-1])

            metrics = meta.data['test_metrics'] 
            if len(metrics) < n_folds:
                for _ in range(n_folds - len(metrics)):
                    metrics.append({})

            # Retrieve training args from the existing metadata
            for k, v in meta.data['args'].items():
                args.__setattr__(k, v)
        args.checkpoint_path = chkpt_path

    return meta

def create_metrics(ignore_index):
    step_metrics = MetricCollection({
        'f1' : F1Score(task='binary', ignore_index=ignore_index),
        'precision' : Precision(task='binary',ignore_index=ignore_index),
        'recall' : Recall(task='binary', ignore_index=ignore_index),
    })

    epoch_metrics = MetricCollection({
        'f1' : F1Score(task='binary', ignore_index=ignore_index),
        'precision' : Precision(task='binary',ignore_index=ignore_index),
        'recall' : Recall(task='binary', ignore_index=ignore_index),
        'auroc' : AUROC('binary', ignore_index=ignore_index),
        'auprc' : AveragePrecision('binary', ignore_index=ignore_index),
        'mcc' : MatthewsCorrCoef('binary', ignore_index=ignore_index)
    })
    
    return step_metrics, epoch_metrics

def compute_averages(meta : Metadata, verbose=True):
    if verbose:
        print('Overall test metric averages')

    buffer = {k : 0 for k in meta.data['test_metrics'][-1].keys()}
    for fold in range(len(meta.data['test_metrics'])):
        fold_metrics = meta.data['test_metrics'][fold]
        for k, v in fold_metrics.items():
            buffer[k] += v

    for k, v in buffer.items():
        buffer[k] = v / len(meta.data['test_metrics'])
        if verbose:
            print(f'mean {k} : {buffer[k]}')

    meta.data['test_metric_avg'] = buffer


def run_training(args : Namespace, create_model_fn):
    if args.release:
        run_release_training(args, create_model_fn)
        return
    
    L.seed_everything(args.seed)

    log_dirname = args.o if args.o else "{}_{}".format(
            os.path.basename(globals().get("__file__", "notebook")),
            datetime.datetime.now().strftime("%Y_%m_%d_%H%M"),
        )

    args.logdir = os.path.join("new_logs", log_dirname)

    meta = handle_metadata(args)
    tokenizer = get_tokenizer(args)
    full_dataset = prepare_datasets(args, ignore_label=args.ignore_label)


    master_logdir = args.logdir

    if args.fold is not None:
        # Single fold training
        start, end = args.fold, args.fold + 1

    else:
        start, end = meta.data['current_fold'], full_dataset.n_splits

    for fold in range(start, end):

        print(f'Current fold: {fold}')
        train_ds, dev_ds, test_ds = full_dataset.get_fold(fold)
        
        train = DataLoader(train_ds, args.batch_size, shuffle=True,
                            collate_fn=partial(prep_batch, tokenizer=tokenizer, ignore_label=args.ignore_label,
                                               modify_prob=args.modify_prob, mask_prob=args.mask_prob, rand_prob=args.rand_prob,
                                               sub_prob=args.sub_prob, kinases=args.kinase),
                            persistent_workers=True if args.num_workers > 0 else False, 
                            num_workers=args.num_workers )
        
        dev_collate_fn = partial(prep_batch, tokenizer=tokenizer, ignore_label=args.ignore_label, modify_prob=0, kinases=args.kinase)
        # Do not modify input residues when not training
        dev = DataLoader(dev_ds, args.batch_size, shuffle=False,
                            collate_fn=dev_collate_fn,
                            persistent_workers=True if args.num_workers > 0 else False,
                            num_workers=args.num_workers)
        
        test = DataLoader(test_ds, args.batch_size, shuffle=False,
                            collate_fn=dev_collate_fn,
                            persistent_workers=True if args.num_workers > 0 else False,
                            num_workers=args.num_workers)
        
        model, tokenizer = prepare_model(args, create_model_fn)

        logdir = os.path.join(master_logdir, f'fold_{fold}')
        model, test_metrics = train_model(args, train, dev, test, model, logdir, fold=fold,
                                                   metadata=meta, master_logdir=master_logdir)
        meta.data['test_metrics'][fold]= test_metrics[0]

        print(f'Test metrics for fold {fold}')
        print(meta.data['test_metrics'][fold])

        meta.data['current_fold'] = fold + 1

        if args.checkpoint_path:
            # Clear the checkpoint after resuming
            args.checkpoint_path = None
        meta.save(master_logdir)
    if args.fold is None:
        compute_averages(meta)
    meta.save(master_logdir)
    return model


def run_release_training(args, create_model_fn):
    L.seed_everything(args.seed)

    log_dirname = args.o if args.o else "{}_{}".format(
            os.path.basename(globals().get("__file__", "notebook")),
            datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S"),
        )

    args.logdir = os.path.join("new_logs", log_dirname)

    meta = handle_metadata(args)
    tokenizer = get_tokenizer(args)
    full_dataset = prepare_full_dataset(args, ignore_label=args.ignore_label)
    step_metrics, epoch_metrics = create_metrics(args.ignore_label)

    master_logdir = args.logdir
    logdir = os.path.join(master_logdir, f'_release')
    model, tokenizer = prepare_model(args, create_model_fn)
    train = DataLoader(full_dataset, args.batch_size, shuffle=True,
                            collate_fn=partial(prep_batch, tokenizer=tokenizer, ignore_label=args.ignore_label, modify_prob=args.modify_prob,
                                                mask_prob=args.mask_prob, sub_prob=args.sub_prob, rand_prob=args.rand_prob),
                            persistent_workers=True if args.num_workers > 0 else False, 
                            num_workers=args.num_workers )
    
    if not isinstance(model, LightningWrapper):
            model = LightningWrapper(args, model, step_metrics=step_metrics, epoch_metrics=epoch_metrics, ds_size=len(train), logdir=logdir)

    model.to(device)
    train_release_model(args, logdir, model, train)

def train_release_model(args, logdir, model, train, dev=None):
    logger = TensorBoardLogger(logdir, name=f'tb_log')
    # Training checkpoint (because having a defined ModelCheckpoint overrides the default checkpointing)
    # Best model checkpoint
    best_callback = ModelCheckpoint(logdir, filename='best', monitor='val_f1', mode='max',
                                    save_on_train_epoch_end=True, auto_insert_metric_name=True)

    es_callback = EarlyStopping('val_f1', patience=args.patience, mode="max")
    callbacks : list[Callback] = [es_callback, best_callback]

    # Use deepspeed 
    if torch.cuda.device_count() > 1:
        strategy = "deepspeed_stage_2"
    else:
        strategy = "auto"
    trainer = L.Trainer(logger=logger, callbacks=callbacks, max_epochs=args.epochs,
                        deterministic=True, log_every_n_steps=1,  accumulate_grad_batches=args.accum, strategy=strategy,
                        default_root_dir=logdir)
    trainer.fit(model, train, dev, ckpt_path=args.checkpoint_path)
    best = torch.load(f'{logdir}/best.ckpt')
    model.load_state_dict(best['state_dict'])
    final_metrics = trainer.test(model, dev)

    # Save predictions into a DataFrame
    pred_df = pd.DataFrame.from_records(model.test_preds, columns=['logits', 'labels', 'sequence_indices', 'df_index'])
    pred_df.to_json(f"{logdir}/release_dev_preds.json")
    print(final_metrics)

    return model, final_metrics