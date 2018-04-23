from WhiteSnake.Vocabulary import Dataset
from config import Env
from tqdm import tqdm
from model.Word2Vec import Word2Vec, SkipGram
from torch.optim import Adam
from Visualization import CustomVisdom
from timeit import timeit
from CustomIO import CSV
import torch
import torch.utils.data as Data
import numpy as np
import multiprocessing
import threading

options = Env()
log = CSV(options.log_name)

def skipgram(**kwargs):
    for k_, v_ in kwargs.items():
        setattr(options, k_, v_)

    vis = CustomVisdom(name=f'skipgram')
    configSummary = ''
    
    for key, value in options.__dict__.items():
        if not key.startswith('__'): configSummary += str(key) + '=' + str(value) + '<br>'
    vis.text('config', f'{configSummary}')

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
    word2Vec = Word2Vec(vocab_size=len(whiteSnake.labelEncoder.classes_), embedding_size=options.embedding_size)
    sgns = SkipGram(embedding=word2Vec, vocab_size=len(whiteSnake.labelEncoder.classes_))
    optim = Adam(sgns.parameters())

    for epoch in tqdm(range(options.epochs)):
        totalLoss = 0

        for index, (batchX, batchY) in tqdm(enumerate(loader)):
            
            if options.use_gpu:
                sgns = sgns.cuda()

            loss = sgns(batchX, batchY)
            totalLoss += loss.data[0]
            optim.zero_grad()
            loss.backward()
            optim.step()

            vis.text('progress', f'目前迭代進度:<br>epochs={epoch + 1}<br>batch={index + 1}')

        
        tqdm.write(f'epochs = {epoch + 1}, loss: {str(totalLoss / options.batch_size)}')
        vis.drawLine('loss', totalLoss / options.batch_size)
        
        log.write([[str(epoch), str(totalLoss / options.batch_size)]])
        torch.save(sgns.state_dict(), f'log/model-{totalLoss / options.batch_size}.pt')
        
    np.savetxt('result/word2vec.txt', word2Vec.ivectors.weight.data.cpu().numpy())
    np.save('result/word2vec', word2Vec.ivectors.weight.data.cpu().numpy())

if __name__ == '__main__':
    import fire
    fire.Fire()
