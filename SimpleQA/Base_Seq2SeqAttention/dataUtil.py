import sys
if sys.platform == "win32":
    from SimpleQA.Base_Seq2SeqAttention.const import *
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
        dataSeqIn = [normalizeString(line.strip()).split(' ') for line in fr.readlines()]
    '''seq.out'''
    with open(pathSeqOut, 'r', encoding='utf-8') as fr:
        dataSeqOut = [line.strip().split(' ') for line in fr.readlines()]
    '''label'''
    with open(pathLabel, 'r', encoding='utf-8') as fr:
        dataLabel = [line.strip() for line in fr.readlines()]
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
    word2index = {word: index + COUNTWSIGN for index, word in enumerate(setWord)}
    word2index["<UNK_WORD>"] = WUNK_SIGN
    word2index["<PAD_WORD>"] = WPAD_SIGN
    word2index["<EOS_WORD>"] = WEOS_SIGN
    index2word = {word2index[word]: word for word in word2index.keys()}
    dictWord   = (word2index, index2word)
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
    slot2index = {slot: index + COUNTSSIGN for index, slot in enumerate(setSlot)}
    slot2index["<UNK_SLOT>"] = SUNK_SIGN
    slot2index["<PAD_SLOT>"] = SPAD_SIGN
    index2slot = {slot2index[slot]: slot for slot in slot2index.keys()}
    dictSlot   = (slot2index, index2slot)
    return dictSlot

def getIntentDictionary(dataLabel):
    """
    根据意图标签数据获取意图字典
    :param dataLabel:
    :return:
    """
    setIntent    = {intent for intent in dataLabel}
    intent2index = {intent: index + COUNTISIGN for index, intent in enumerate(setIntent)}
    intent2index["<UNK_INTENT>"] = IUNK_SIGN
    index2intent = {intent2index[intent]: intent for intent in intent2index.keys()}
    dictIntent   = (intent2index, index2intent)
    return dictIntent



""" 数据预处理 """
def normalizeString(s):
    """
    去掉非目标语言字符（保留特定标点符号）
    :param s:
    :return:
    """
    s = re.sub(r"([.!?])", r"", s)
    s = re.sub(r"[^0-9a-zA-Z.!?\-_\' ]+", r" ", s)
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

        assert len(itemSeqIn) == len(itemSeqOut)
        pairs.append([itemSeqIn, itemSeqOut, itemLabel])

    return pairs


def transIds(pairs, word2index, slot2index, intent2index):
    """
    将字词数据都转换为整数id
    :param pairs:
    :param dictWord:
    :param dictSlot:
    :param dictIntent:
    :return:
    """
    pairsIded = []
    for pair in pairs:
        itemSeqIn  = pair[0]
        itemSeqOut = pair[1]
        itemIntent = pair[2]

        itemSeqInIded  = [word2index.get(word, WUNK_SIGN) for word in itemSeqIn]    # words to ids
        itemSeqOutIded = [slot2index.get(slot, SUNK_SIGN) for slot in itemSeqOut]   # slots to ids
        itemIntentIded  = intent2index.get(itemIntent, 0)                      # labels to ids

        assert len(itemSeqInIded) == len(itemSeqOutIded)
        pairsIded.append([itemSeqInIded, itemSeqOutIded, itemIntentIded])

    return pairsIded


def splitData(pairs, batchSize=BATCHSIZE):
    """
    根据BATCHSIZE讲样例集切分成多个batch
    :param pairs:
    :return:
    """
    random.shuffle(pairs)
    trainIterator = []
    for start in range(0, len(pairs), batchSize):
        trainIterator.append([[item[0] for item in pairs[start:start + batchSize]],
                              [item[1] for item in pairs[start:start + batchSize]],
                              [item[2] for item in pairs[start:start + batchSize]]])
    return trainIterator

def getMaxLengthFromBatch(batch, addLength):
    """
    :param batch: 样例对集合
    :return: Batch中最长的输入序列的长度
    """
    return max(MAXLEN, max([len(seqIn) for seqIn in batch[0]])) + addLength


def getSeqInLengthsFromBatch(batch, addLength, MAXLEN=MAXLEN):
    """
    :param batch: 样例对集合
    :return: []: 所有输入序列的长度
    """
    return [min(MAXLEN, len(seqIn) + addLength) for seqIn in batch[0]]


def padBatch(batch, addLength, MAXLEN_TEMP=MAXLEN):
    """
    根据序列最大长度对数据进行裁剪和pading操作
    :param pairsIded: 样例对
    :return:
    """
    batch[0]       = [item[:MAXLEN_TEMP - addLength] for item in batch[0]]
    batch[1]       = [item[:MAXLEN_TEMP - addLength] for item in batch[1]]

    batchSeqIn     = [(batch[0][index] + [WEOS_SIGN] + [WPAD_SIGN] * MAXLEN_TEMP)[:MAXLEN_TEMP] for index in range(len(batch[0]))]
    batchSeqOut    = [(batch[1][index] + [SPAD_SIGN] + [SPAD_SIGN] * MAXLEN_TEMP)[:MAXLEN_TEMP] for index in range(len(batch[1]))]
    trainIterator = [batchSeqIn, batchSeqOut, batch[2]]

    return trainIterator

def vector2Tensor(BatchSeqIn, BatchSeqOut, BatchLabel):
    BatchSeqIn  = torch.tensor(BatchSeqIn, dtype=torch.long).cuda() if torch.cuda.is_available() else torch.tensor(BatchSeqIn, dtype=torch.long, device="cpu")
    BatchSeqOut = torch.tensor(BatchSeqOut, dtype=torch.long).cuda() if torch.cuda.is_available() else torch.tensor(BatchSeqOut, dtype=torch.long, device="cpu")
    BatchLabel  = torch.tensor(BatchLabel, dtype=torch.long).cuda() if torch.cuda.is_available() else torch.tensor(BatchLabel, dtype=torch.long, device="cpu")

    return BatchSeqIn, BatchSeqOut, BatchLabel
