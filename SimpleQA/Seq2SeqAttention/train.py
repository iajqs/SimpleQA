import sys
print(sys.platform)
if sys.platform == "win32":
    from SimpleQA.Seq2SeqAttention.model import *
else:
    from model import *

import re
import random
import torch
import torch.nn as nn
import torch.optim as optim

""" 路径 """
if sys.platform == "win32":
    trainDir = "H:/研究生/code/SimpleQA/data/atis/train"
    validDir = "H:/研究生/code/SimpleQA/data/atis/valid"
    testDir = "H:/研究生/code/SimpleQA/data/atis/test"
else:
    trainDir = "/home/cks/program/SimpleQA/data/atis/train"
    validDir = "/home/cks/program/SimpleQA/data/atis/valid"
    testDir = "/home/cks/program/SimpleQA/data/atis/test"


""" 设定模型超参数 """
TRAINITER    =   20         # 迭代训练次数#
BATCHSIZE    =  128         # 切分出来的每个数据块的大小
PAD_IDX      =    0         # pad 在词典的下标
EOS_IDX      =    1         # 结束符下标
WORDSIZE     =    0         # 词典大小
SLOTSIZE     =    0         # 词槽字典大小
INTENTSIZE   =    0         # 意图字典大小
EMBEDDSIZE   =  128         # 词向量大小
LSTMHIDSIZE  =  128         # LSTM隐含层大小
NLAYER       =    1         # LSTM的层数
DROPOUT      =  0.1         # dropout系数
MAXLEN       =   20         # 序列最大长度（训练时）
CLIP         =    1         # 最大梯度值
LEARNINGRATE = 1e-3         # 学习速率


""" 读取数据 """
def getData(dataDir):
    """
    读取目录下的数据：seq.in, seq.out, label
    :param path:
    :return:
    """
    pathSeqIn  = dataDir + "/seq.in"          # 输入序列文件路径
    pathSeqOut = dataDir + "/seq.out"         # 输出序列文件路径
    pathLabel  = dataDir + "/label"           # 输出意图文件路径

    dataSeqIn  = []      # 输入序列数据
    dataSeqOut = []      # 输出序列数据
    dataLabel  = []      # 意图标签数据

    '''seq.in'''
    with open(pathSeqIn, 'r', encoding='utf-8') as fr:
        dataSeqIn = [normalizeString(line.strip()).split() for line in fr.readlines()]
    '''seq.out'''
    with open(pathSeqOut, 'r', encoding='utf-8') as fr:
        dataSeqOut = [normalizeString(line.strip()).split() for line in fr.readlines()]
    '''label'''
    with open(pathLabel, 'r', encoding='utf-8') as fr:
        dataLabel = [line.strip() for line in fr.readlines()]

    assert len(dataSeqIn) == len(dataSeqOut) == len(dataLabel)

    return dataSeqIn, dataSeqOut, dataLabel

""" 获取词典 """
def getWordDictionary(dataSeqin):
    """
    根据输入序列数据获取词槽标签字典
    :param dataSeqin:
    :return:
    """
    setWord = set()
    for line in dataSeqin:
        for word in line:
            setWord.add(word)
    word2index = {word: index + 2 for index, word in enumerate(setWord)}
    word2index["<PAD>"] = 0
    word2index["<UNK>"] = 1
    index2word = {word2index[word]: word for word in word2index.keys()}
    dictWord = (word2index, index2word)
    return dictWord

""" 获取标签字典 """
def getSlotDictionary(dataSeqOut):
    """
    根据输出序列数据获取词槽标签字典
    :param dataSeqOut:
    :return:
    """
    setSlot = set()
    for line in dataSeqOut:
        for slot in line:
            setSlot.add(slot)
    slot2index = {slot: index + 1 for index, slot in enumerate(setSlot)}
    slot2index["<PAD>"] = 0
    index2slot = {slot2index[slot]: slot for slot in slot2index.keys()}
    dictSlot = (slot2index, index2slot)
    return dictSlot

def getIntentDictionary(dataLabel):
    """
    根据意图标签数据获取意图字典
    :param dataLabel:
    :return:
    """
    setLabel = {label for label in dataLabel}
    label2index = {label: index + 1 for index, label in enumerate(setLabel)}
    label2index[0] = "<UNK_LABEL>"
    index2label = {label2index[label]: label for label in label2index.keys()}
    dictLabel = (label2index, index2label)
    return dictLabel



""" 数据预处理 """
def normalizeString(s):
    """
    去掉非目标语言字符（保留特定标点符号）
    :param s:
    :return:
    """
    s = re.sub(r"([.!?])", r" .", s)
    s = re.sub(r"[^0-9a-zA-Z.!?]+", r" ", s)
    return s

def makePairs(dataSeqIn, dataSeqOut, dataLabel):
    """
    根据读取的数据生成样例对
    :param dataSeqIn: 输入序列
    :param dataSeqOut: 词槽标签序列
    :param dataLabel: 意图标签
    :return: pairs: zip(dataSeqIn, dataSeqOut, dataLabel)
    """
    size = len(dataSeqIn)

    pairs = []
    for index in range(size):
        itemSeqIn  = dataSeqIn[index]
        itemSeqOut = dataSeqOut[index]
        itemLabel  = dataLabel[index]

        pairs.append([itemSeqIn, itemSeqOut, itemLabel])

    return pairs


def transIds(pairs, word2index, slot2index, label2index):
    """
    将字词数据都转换为整数id
    :param pairs:
    :param dictWord:
    :param dictSlot:
    :param dictLabel:
    :return:
    """
    pairsIded = []
    for pair in pairs:
        itemSeqIn  = pair[0]
        itemSeqOut = pair[1]
        itemLabel  = pair[2]

        itemSeqInIded  = [word2index.get(word, 2) for word in itemSeqIn]    # words to ids
        itemSeqOutIded = [slot2index[slot] for slot in itemSeqOut]          # slots to ids
        itemLabelIded  = label2index.get(itemLabel, 0)                      # labels to ids

        pairsIded.append([itemSeqInIded, itemSeqOutIded, itemLabelIded])

    return pairsIded


def pad(pairsIded):
    """
    根据序列最大长度对数据进行裁剪和pading操作
    :param pairsIded: 样例对
    :return:
    """
    pairsIdedPaded = []
    for pair in pairsIded:
        itemSeqIn  = pair[0]
        itemSeqOut = pair[1]
        itemLabel  = pair[2]

        itemSeqIn  = (itemSeqIn + [PAD_IDX] * MAXLEN)[:MAXLEN]
        itemSeqOut = (itemSeqOut + [PAD_IDX] * MAXLEN)[:MAXLEN]

        pairsIdedPaded.append([itemSeqIn, itemSeqOut, itemLabel])

    return pairsIdedPaded


def splitData(pairs):
    """
    根据BATCHSIZE讲样例集切分成多个batch
    :param pairs:
    :return:
    """
    random.shuffle(pairs)
    trainIterator = []
    for start in range(0, len(pairs), BATCHSIZE):
        trainIterator.append([[item[0] for item in pairs[start:start+BATCHSIZE]],
                              [item[1] for item in pairs[start:start + BATCHSIZE]],
                              [item[2] for item in pairs[start:start + BATCHSIZE]]])
    return trainIterator

def vector2Tensor(BatchSeqIn, BatchSeqOut, BatchLabel):
    BatchSeqIn  = torch.tensor(BatchSeqIn, dtype=torch.long, device="cpu")
    BatchSeqOut = torch.tensor(BatchSeqOut, dtype=torch.long, device="cpu")
    BatchLabel  = torch.tensor(BatchLabel, dtype=torch.long, device="cpu")

    return BatchSeqIn, BatchSeqOut, BatchLabel


""" 初始化模型 """
def init_weights(model):
    for name, param in model.parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

def initModel(WORDSIZE, SLOTSIZE, INTENTSIZE):
    """
    初始化所有模型原件，并将原件传给总模型框架
    EncoderRNN、AttnIntent、DecoderIntent -> Seq2Intent
    EncoderRNN、AttnSlot、DecoderSlot -> Seq2Slot
    Seq2Intent、Seq2Slot -> Seq2Seq
    :return:
    """
    encoder       = EncoderRNN(input_size=WORDSIZE, emb_size=EMBEDDSIZE, hidden_size=LSTMHIDSIZE, n_layers=NLAYER, dropout=DROPOUT)
    attnIntent    = AttnIntent()
    decoderIntent = DecoderIntent(hidden_size=LSTMHIDSIZE, intent_size=INTENTSIZE)
    model = Seq2Intent(encoder, dec_intent=decoderIntent, attn_intent=attnIntent, hidden_size=LSTMHIDSIZE)
    #model.apply(init_weights)
    return model


""" 设定模型优化器 """
def initOptimize(model):
    return optim.Adam(model.parameters(), lr=LEARNINGRATE)

""" 设定损失函数 """
def initLossFunction(PAD_IDX=None):
    return nn.CrossEntropyLoss(ignore_index=PAD_IDX)

""" 训练 """
def train(iter, model=None):

    ''' 读取数据 '''
    dataSeqIn, dataSeqOut, dataLabel = getData(trainDir)                 # 获取原数据
    dictWord  = getWordDictionary(dataSeqIn)                             # 获取词典  (word2index, index2word)
    dictSlot  = getSlotDictionary(dataSeqOut)                            # 获取词槽标签字典  (slot2index, index2slot)
    dictLabel = getIntentDictionary(dataLabel)                           # 获取意图标签字典  (label2index, index2label)
    pairs = makePairs(dataSeqIn, dataSeqOut, dataLabel)                  # 根据原数据生成样例对    zip(dataSeqIn, dataSeqOut, dataLabel)
    pairsIded = transIds(pairs, dictWord[0], dictSlot[0], dictLabel[0])  # 将字词都转换为数字id
    pairsIdedPaded = pad(pairsIded)                                      # 对数据进行pad填充与长度裁剪
    trainIterator = splitData(pairsIdedPaded)                            # 讲样例集按BATCHSIZE大小切分成多个块


    ''' 设定字典大小参数 '''
    WORDSIZE   = len(dictWord[0])
    SLOTSIZE   = len(dictSlot[0])
    INTENTSIZE = len(dictLabel[0])

    ''' 定义模型、优化器、损失函数 '''
    model = initModel(WORDSIZE, SLOTSIZE, INTENTSIZE) if model == None else model # 初始化并返回模型

    optimizer = initOptimize(model)                   # 初始化并返回优化器
    criterion = initLossFunction(PAD_IDX)             # 初始化并返回损失函数

    ''' 模型训练 '''
    model.train()                                   # 设定模型状态为训练状态
    epoch_loss = 0                                  # 定义总损失

    for epoch, batch in enumerate(trainIterator):
        BatchSeqIn  = batch[0]
        BatchseqOut = batch[1]
        Batchlabel  = batch[2]
        BatchSeqIn, BatchseqOut, Batchlabel = vector2Tensor(BatchSeqIn, BatchseqOut, Batchlabel)

        optimizer.zero_grad()

        # output = model(BatchSeqIn, BatchseqOut)
        output       = model(BatchSeqIn)
        # outputSeqOut = output.seqOut
        outputLabel  = output

        lossLabel = criterion(outputLabel, Batchlabel)

        loss = lossLabel
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()

        epoch_loss += loss.item()
        print("iter=%d, epoch=%d / %d: trainLoss = %f" % (iter, epoch, len(trainIterator), epoch_loss / (epoch + 1)))
    return epoch_loss / len(trainIterator), model, (dictWord, dictSlot, dictLabel)

def evaluate(model, dicts):

    ''' 读取数据 '''
    dataSeqIn, dataSeqOut, dataLabel = getData(validDir)  # 获取原数据
    dictWord = dicts[0]  # 获取词典  (word2index, index2word)
    dictSlot = dicts[1]  # 获取词槽标签字典  (slot2index, index2slot)
    dictLabel = dicts[2]  # 获取意图标签字典  (label2index, index2label)
    pairs = makePairs(dataSeqIn, dataSeqOut, dataLabel)  # 根据原数据生成样例对    zip(dataSeqIn, dataSeqOut, dataLabel)
    pairsIded = transIds(pairs, dictWord[0], dictSlot[0], dictLabel[0])  # 将字词都转换为数字id
    pairsIdedPaded = pad(pairsIded)  # 对数据进行pad填充与长度裁剪
    validIterator = splitData(pairsIdedPaded)  # 讲样例集按BATCHSIZE大小切分成多个块

    criterion = initLossFunction(PAD_IDX)           # 初始化并返回损失函数

    ''' 模型验证 '''
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(validIterator):
            BatchSeqIn = batch[0]
            BatchseqOut = batch[1]
            Batchlabel = batch[2]
            BatchSeqIn, BatchseqOut, Batchlabel = vector2Tensor(BatchSeqIn, BatchseqOut, Batchlabel)

            output = model(BatchSeqIn)
            outputLabel = output

            lossLabel = criterion(outputLabel, Batchlabel)
            epoch_loss += lossLabel.item()

    return epoch_loss / len(validIterator)


if __name__ == '__main__':
    import time
    model = None
    for iter in range(TRAINITER):
        # print("-" * 20, epoch, "-" * 20)
        trainLoss, model, dicts = train(iter, model)

        # print()

        # time.sleep(100)
        validLoss = evaluate(model, dicts)
        print("iter %d / %d: trainLoss = %f, validLoss = %f" % (iter, TRAINITER, trainLoss, validLoss))