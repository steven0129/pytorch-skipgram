from WhiteSnake.Vocabulary import Dataset
from config import Env
from tqdm import tqdm
from model.Word2Vec import Word2Vec, SkipGram
import torch
import torch.utils.data as Data
from torch.optim import Adam
import numpy as np
import multiprocessing

options = Env()


def skipgram(**kwargs):
    for k_, v_ in kwargs.items():
        setattr(options, k_, v_)

    whiteSnake = Dataset(ratio=options.ratio, windowSize=options.window_size)
    print('將每對pair分別存入X與Y...')
    pool = multiprocessing.Pool(4)
    XY = pool.map(list, tqdm(whiteSnake))
    
    X = list(list(zip(*XY))[0])
    Y = list(list(zip(*XY))[1])

    X = torch.Tensor(X).long()
    Y = torch.Tensor(Y).long()

    dataset = Data.TensorDataset(
        data_tensor=X,
        target_tensor=Y)
    loader = Data.DataLoader(
        dataset=dataset,
        batch_size=options.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=options.core)

    print('開始進行迭代...')
    for epoch in tqdm(range(options.epochs)):
        totalLoss = 0

        for batchX, batchY in tqdm(loader):
            word2Vec = Word2Vec(vocab_size=len(whiteSnake.labelEncoder.classes_), embedding_size=500)
            sgns = SkipGram(embedding=word2Vec, vocab_size=len(whiteSnake.labelEncoder.classes_))
            optim = Adam(sgns.parameters())

            if options.use_gpu:
                sgns = sgns.cuda()

            loss = sgns(batchX, batchY)
            totalLoss += loss.data[0]
            optim.zero_grad()
            loss.backward()
            optim.step()

        
        tqdm.write(f'epochs = {epoch + 1}, loss: {str(totalLoss)}')


if __name__ == '__main__':
    import fire
    fire.Fire()
