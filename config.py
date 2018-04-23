class Env(object):
    ratio = 1 # 訓練集比例
    batch_size = 50 # batch size
    shuffle = True # 打亂data
    core = 1 # 使用核心數
    epochs = 50 # epoch
    use_gpu = False # 是否使用GPU
    window_size = 4 # 設定Word2Vec的Window Size
    log_name = 'loss' # 設定存放紀錄的檔案名稱
