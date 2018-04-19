from WhiteSnake.Vocabulary import Dataset
from config import Env
from tqdm import tqdm
from model.Word2Vec import Word2Vec, SkipGram
import torch
import torch.utils.data as Data
from torch.optim import Adam
import numpy as np
import multiprocessing
import visdom

options = Env()


def skipgram(**kwargs):
    for k_, v_ in kwargs.items():
        setattr(options, k_, v_)

    vis = visdom.Visdom()
    whiteSnake = Dataset(ratio=options.ratio, windowSize=options.window_size)
    print('將每對pair分別存入X與Y...')
    pool = multiprocessing.Pool()

    X = []
    Y = []
    progress = 0
    win = vis.text(f'目前資料輸入進度: {progress}/{len(whiteSnake)}')
    for XY in tqdm(pool.imap_unordered(list, whiteSnake), total=len(whiteSnake)):
        X.append(XY[0])
        Y.append(XY[1])
        progress += 1
        vis.text(f'目前資料輸入進度: {progress}/{len(whiteSnake)}', win=win)
        
    
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
