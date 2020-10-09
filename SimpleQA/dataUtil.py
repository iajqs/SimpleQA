import sys
if sys.platform == "win32":
    from SimpleQA.const import *
else:
    from .const import *

import re
import random
import torch


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
