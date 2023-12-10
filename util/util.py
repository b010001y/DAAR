import random
import torch
import numpy as np

def set_seed(config, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if config['n_gpu'] > 0 and torch.cuda.is_available() and config['use_gpu']:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False