from WhiteSnake.Vocabulary import Dataset
from config import Env
from tqdm import tqdm
from model.Word2Vec import Word2Vec, SkipGram
from torch.optim import Adam
from utils.Visualization import CustomVisdom
from utils.IO import CSV
import torch
import torch.utils.data as Data
import numpy as np
import multiprocessing

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
    print(f'收錄{len(whiteSnake.labelEncoder.classes_)}個字...')
    print('將每對pair分別存入X與Y...')
    pool = multiprocessing.Pool()

    X = [0] * len(whiteSnake)
    Y = [0] * len(whiteSnake)

    for i, (x, y) in pool.imap_unordered(tuple, tqdm(enumerate(whiteSnake), total=len(whiteSnake)), chunksize=100):
        X[i] = x
        Y[i] = y
        vis.text('progress', f'目前資料輸入進度: {i + 1}/{len(whiteSnake)}')
        
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
        batchNum = int(len(whiteSnake) / options.batch_size)

        for index, (batchX, batchY) in enumerate(tqdm(loader)):
            
            if options.use_gpu:
                sgns = sgns.cuda()

            loss = sgns(batchX, batchY)
            totalLoss += loss.data[0]
            optim.zero_grad()
            loss.backward()
            optim.step()

            vis.text('progress', f'目前迭代進度:<br>epochs={epoch + 1}<br>batch={index + 1}')

        avgLoss = totalLoss / batchNum
        tqdm.write(f'epochs = {epoch + 1}, loss = {str(avgLoss)}')
        vis.drawLine('loss', x=epoch + 1, y=avgLoss)
        
        log.write([[str(epoch + 1), str(avgLoss)]])
        torch.save(sgns.state_dict(), f'log/model/model-{avgLoss}.pt')
        
    np.savetxt('log/result/word2vec.txt', word2Vec.ivectors.weight.data.cpu().numpy())
    np.save('log/result/word2vec.npy', word2Vec.ivectors.weight.data.cpu().numpy())
    np.save('log/result/classes.npy', whiteSnake.labelEncoder.classes_)

if __name__ == '__main__':
    import fire
    fire.Fire()
