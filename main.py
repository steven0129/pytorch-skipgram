from WhiteSnake.Vocabulary import Dataset
from config import Env
from tqdm import tqdm
import torch
import torch.utils.data as Data

options = Env()


def train(**kwargs):
    for k_, v_ in kwargs.items():
        setattr(options, k_, v_)

    whiteSnake = Dataset(ratio=options.ratio, windowSize=options.window_size)
    X = []
    Y = []
    for pairs in whiteSnake:
        for pair in list(pairs):
            X.append(pair[0])
            Y.append(pair[1])

    dataset = Data.TensorDataset(
        data_tensor=torch.Tensor(X),
        target_tensor=torch.Tensor(Y))
    loader = Data.DataLoader(
        dataset=dataset,
        batch_size=options.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=options.core)

    for epoch in tqdm(range(options.epochs)):
        for batchX, batchY in loader:
            tqdm.write(str(batchX.numpy()) + ', ' + str(batchY.numpy()))


if __name__ == '__main__':
    import fire
    fire.Fire()
