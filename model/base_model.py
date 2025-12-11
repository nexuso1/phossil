# This is a base class that all classifiers inherit. It 
# manages the loss calculation, as well as other functions. 

import torch
from typing import Callable
from dataclasses import dataclass

@dataclass
class BaseModelConfig:
    loss : Callable[[torch.Tensor], torch.Tensor]|dict[str, Callable[[torch.Tensor], torch.Tensor]]
    ignore_index : int = -1
    logger : Callable = None

class BaseModel(torch.nn.Module):
    def __init__(self, config : BaseModelConfig) -> None:
        super(BaseModel, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.logger = config.logger
        self.ignore_index = config.ignore_index
        self.loss = config.loss

    def save(self, path):
        trainable_state_dict = {
            name: param.data
            for name, param in self.named_parameters()
            if param.requires_grad
        }
        
        torch.save({
            'state_dict' : trainable_state_dict,
            'config' : self.config
        }, path)

        print(f'Trainable weights and config saved to {path}')

    def load(self, path):
        saved_model = torch.load(path)
        print(f'Loading state dict from {path}...')
        keys = self.load_state_dict(saved_model['state_dict'], strict=False)
        print(f'Finished loading. Incompatible keys (not necessarily an error): {keys}')
        self.config = saved_model['config']

    def init_weights(self, module):
        for param in module.parameters():
            self.xavier_init(param)
            
    def xavier_init(self, m):
        """
        Uses xavier/glorot weight initialization for linear layers, and 0 for bias
        """
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0)

    def set_base_requires_grad(self, requires_grad : bool):
        """
        Freeze/unfreeze the base model according to the requires_grad bool
        """
        for p in self.base.parameters():
          p.requires_grad = requires_grad

    def compute_loss(self, inputs, outputs, **kwargs):
        """
        Compute the loss of outputs w.r.t. model weights and inputs.
        
        :param inputs: Model inputs
        :param outputs: Model outputs
        :param kwargs: Other arguments from the inputs
        """
        raise NotImplementedError('Need to implement loss computation')

    def predict(self, inputs, *args, **kwargs) -> torch.Tensor:
        """
        Prediction in eval mode.
        Outputs are the final classification logits as a Tensor.
        """
        self.eval()
        with torch.no_grad():
            return self(inputs, *args, **kwargs)

    def train_predict(self, inputs, *args, **kwargs):
        """
        Prediction in train mode. Labels should be provided.
        Outputs will contain the loss w.r.t inputs.

        If return_dict is True, output will be a dictionary with additional information about hidden
        states and attentions of the base model
        """
        self.train()
        outputs = self(inputs, *args, **kwargs)
        loss = self.compute_loss(inputs, outputs, **kwargs)
        return loss, outputs
    
    def forward(self, inputs, **kwargs):
        """
        Forward pass of the model. Does not make any changes to train/eval status, and does
        not calculate loss.

        Returns (logits, outputs of the base model).
        """
        raise NotImplementedError('Need to implement the forward pass')