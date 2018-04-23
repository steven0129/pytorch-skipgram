from WhiteSnake.Vocabulary import Dataset
from config import Env
from tqdm import tqdm
from model.Word2Vec import Word2Vec, SkipGram
from torch.optim import Adam
from Visualization import CustomVisdom
from timeit import timeit
import torch
import torch.utils.data as Data
import numpy as np
import multiprocessing
import threading


options = Env()


def skipgram(**kwargs):
    for k_, v_ in kwargs.items():
        setattr(options, k_, v_)

    vis = CustomVisdom(name='skigram')
    whiteSnake = Dataset(ratio=options.ratio, windowSize=options.window_size)
    print('將每對pair分別存入X與Y...')
    pool = multiprocessing.Pool()

    X = [0] * len(whiteSnake)
    Y = [0] * len(whiteSnake)
    progress = 0

    vis.text('progress', f'目前資料輸入進度: {progress}/{len(whiteSnake)}')
    for i, (x, y) in pool.imap_unordered(tuple, tqdm(enumerate(whiteSnake), total=len(whiteSnake)), chunksize=100):
        X[i] = x
        Y[i] = y
        progress += 1
        vis.text('progress', f'目前資料輸入進度: {progress}/{len(whiteSnake)}')
        
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

        for index, (batchX, batchY) in tqdm(enumerate(loader)):
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

            vis.text('progress', f'目前迭代進度:<br>batch={index}<br>epochs={epoch + 1}')

        
        tqdm.write(f'epochs = {epoch + 1}, loss: {str(totalLoss / options.batch_size)}')
        vis.drawLine('loss', totalLoss / options.batch_size)


if __name__ == '__main__':
    import fire
    fire.Fire()
