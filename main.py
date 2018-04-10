from data import WhiteSnake
from config import Env
import torch
import torch.utils.data as Data

options = Env()

def train(**kwargs):
    for k_,v_ in kwargs.items():
        setattr(options,k_,v_)
    
    whiteSnake = list(WhiteSnake(ratio=options.ratio))
    whiteSnake = list(zip(*whiteSnake))
 
    dataset = Data.TensorDataset(data_tensor=torch.Tensor(whiteSnake[0]), target_tensor=torch.Tensor(whiteSnake[1]))
    loader = Data.DataLoader(dataset=dataset, batch_size=options.batch_size, shuffle=options.shuffle, num_workers=options.core)

    for epoch in range(options.epochs):
        for batchX, batchY in loader:
            print(batchX.size())

if __name__=='__main__':
    import fire
    fire.Fire()