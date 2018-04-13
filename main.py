from WhiteSnake.Vocabulary import Dataset
from config import Env
from tqdm import tqdm
import torch
import torch.utils.data as Data

options = Env()

def train(**kwargs):
    for k_,v_ in kwargs.items():
        setattr(options,k_,v_)
    
    whiteSnake = Dataset(ratio=options.ratio, windowSize=options.window_size)

if __name__=='__main__':
    import fire
    fire.Fire()