import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    dim:int=4096
    n_layers:int=32
    n_heads:int=32 #n.o of head for queries
    n_kv_heads:Optional[int]=None #n.o of heads for k and v
    vocab_size:int=-1 # will be set when tokenizer is loaded
    multiple_of:int=256
    ffn_dim_multiplier:Optional[float]=None
    norm_eps:float=1e-5

    #KV cache

    max_batch_size:int=32
    max_seq_len:int=2048

    device:str=None







#Llama2 embedding -> 4096