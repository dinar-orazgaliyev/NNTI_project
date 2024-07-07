import math
from typing import List, Tuple

import torch
import torch.nn as nn



class LinearLoRA(nn.Module):
    """
    A Low-rank Adapted Linear Layer (LoRA).
    
    Args:
        in_dim (int): Input dimension of the linear layer.
        out_dim (int): Output dimension of the linear layer.
        r (int): Rank of the low-rank approximated matrices.
        lora_alpha (int): Numerator of the scaling constant alpha / r.
        lora_dropout (float): Dropout probability.
    """      
    
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        r: int = 8, 
        lora_alpha: int = 16, 
        lora_dropout: float = 0.,    
    ):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(lora_dropout) 
        
        
        assert r > 0, "Rank must be greater than zero."
            
        # Initializing the pretrained linear layer, actual weights will be copied by the function outside of the class
        self.pretrained = nn.Linear(in_dim, out_dim, bias=True)
        self.pretrained.weight.requires_grad = False

        # create the low-rank A matrix and initialize with gaussian distribution
        self.lora_A = nn.Linear(in_dim, r, bias=False)
        nn.init.normal(self.lora_A.weight, mean=0, std=0.02)

        # create the low-rank B matrix and initialize to zero 
        
        self.lora_B = nn.Linear(r, out_dim, bias=False)
        nn.init.zeros_(self.lora_B.weight)

        # scaling constant
        self.scaling = self.lora_alpha / self.r
                        
    def forward(self, x):
        #here we do the matrix multiplications, lora_dropout @ lora_A @ lora_B * scaling
        pretrained_out = self.pretrained(x)
        lora_out = self.lora_dropout(x)
        lora_out = self.lora_A(lora_out)
        lora_out = self.lora_B(lora_out)
        lora_out = lora_out * self.scaling        
        return pretrained_out + lora_out
    
    
def model_freeze(model):
    """Freeze the originals model's layers except for added lora layers"""
    for name, param in model.named_parameters():
        if "lora" not in name:
            param.requires_grad = False

            
def convert_to_lora(module, r, lora_dropout, lora_alpha):
    """Converts a linear module to a LoRA linear module."""
    k, d = module.weight.shape  
    lora = LinearLoRA(d, k, r, lora_dropout=lora_dropout, lora_alpha=lora_alpha)
    with torch.no_grad():
        lora.pretrained.weight.copy_(module.weight)
        lora.pretrained.bias.copy_(module.bias)        
    return lora
                

                
def add_lora_layers(
    model,  
    r: int=8, 
    lora_alpha: float=16,
    lora_dropout: float=0.1  
):
    """
        Adds the LoRA layers to the model. 
     
        Args:
        model (torch.nn.Module): The PyTorch model to be used.
        r (int): Rank of low-rank approximation.
        lora_alpha (float): Numerator of scaling constant.
        lora_dropout (float): Dropout probability.
           
        """                     
    module_types: Tuple=(nn.Linear,)
    
    # disable dropout in frozen layers
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
    # replace chosen linear modules with lora modules
    for name, module in model.named_children():
        if isinstance(module, module_types) and (('q_proj' in name) or ('v_proj' in name)):
            temp_lora = convert_to_lora(module, r=r, lora_dropout=lora_dropout, lora_alpha=lora_alpha)
            setattr(model, name, temp_lora)                  
        else:
            
            add_lora_layers(module, r, lora_dropout, lora_alpha)               
def model_unfreeze(model):
    """Unfreezes all parameters in a model"""
    for name, param in model.named_parameters():
        param.requires_grad = True

        
def merge_weights(module):
    """Converts a LoRA linear module back to a linear module."""
    k, d = module.pretrained.weight.shape  
    linear = nn.Linear(d, k, bias=True)
    
    with torch.no_grad():
        #The actual merging of the pretrained weights and lora weights
        linear.weight.copy_(module.pretrained.weight + (module.lora_B.weight @ module.lora_A.weight) * module.scaling)
        linear.bias.copy_(module.pretrained.bias)
        
    return linear


def merge_lora_layers(model, module_names: Tuple=("q_proj", "v_proj"), dropout=0.1):
    """
        Replaces LoRA modules with their original linear equivalents. 
   
        Args:
        model (torch.nn.Module): The PyTorch model to be used.
        module_names (Tuple): Names of the LoRA layers to replace.
        dropout (float): Dropout probability.  
    """                     
    # enable dropout in frozen layers
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = dropout
    # replace chosen linear modules with lora modules
    for name, module in model.named_children():
        if name in module_names and hasattr(module, "pretrained"):
            temp_linear = merge_weights(module)
            setattr(model, name, temp_linear)                  
        else:
            merge_lora_layers(module, module_names=module_names, dropout=0.1)
                         