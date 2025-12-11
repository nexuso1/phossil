import torch
from base_model import BaseModel, BaseModelConfig
from dataclasses import dataclass

@dataclass
class LMModelConfig(BaseModelConfig):
    base : str = '650M'
    dropout : float = 0

@dataclass
class LMModelInputs:
    input_ids : torch.Tensor
    attention_mask : torch.Tensor = None
    labels : torch.Tensor = None

@dataclass
class LMModelOuptut:
    logits : torch.Tensor
    base_output = None

class LMModel(BaseModel):
    def __init__(self, config, base : torch.nn.Module):
        super().__init__(config)
        self.base = base

    def compute_loss(self, inputs : LMModelInputs, outputs : LMModelOuptut, return_dict=False, **kwargs):
        logits = outputs.logits
        labels = inputs.labels

        if inputs.attention_mask is not None:
            active_loss = inputs.labels.view(-1) != self.config.ignore_index
            active_logits = logits.reshape(-1, self.n_labels)
            valid_logits=active_logits[active_loss].flatten()
            valid_labels=labels.view(-1)[active_loss]
            loss = self.loss(valid_logits, valid_labels)
            
        else:
            loss = self.loss(logits.view(-1, self.n_labels), labels.view(-1))


        loss += outputs.base_output.loss

        if not return_dict:
            return loss, logits
        
        return {
            'loss' : loss,
            'logits' : logits,
            'hidden_states' : outputs.hidden_states,
            'attentions' : outputs.attentions,
            'outputs' : outputs
        } 
    
    def forward(self, inputs : LMModelInputs, **kwargs):
        outputs = self.base(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        sequence_output = outputs[0]
        classifier_features = self.classifier_features(sequence_output, **kwargs)
        return self.classifier(classifier_features), outputs