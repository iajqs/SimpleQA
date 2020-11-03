import sys
if sys.platform == "win32":
    from .const import *
else:
    from .const import *

import re
import random
import torch


""" const """


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
        dataSeqIn = [["[CLS]"] + normalizeString(line.strip()).split(' ') for line in fr.readlines()]
    '''seq.out'''
    with open(pathSeqOut, 'r', encoding='utf-8') as fr:
        dataSeqOut = [["O"] + line.strip().split(' ') for line in fr.readlines()]
    '''label'''
    with open(pathLabel, 'r', encoding='utf-8') as fr:
        dataLabel = [line.strip() for line in fr.readlines()]

    return dataSeqIn, dataSeqOut, dataLabel

""" 获取词典 """
def getBERTWordDictionary(tokenizer):
    """
    根据输入序列数据获取词槽标签字典
    :param dataSeqin:
    :return:
    """
    word2index = {}
    for token in tokenizer.vocab.items():
        if token[1] < COUNTWSIGN:
            continue
        word2index[token[0]] = token[1]
    word2index["[UNK]"] = WBUNK_SIGN          # <UNK_WORD>
    word2index["[PAD]"] = WPAD_SIGN          # <PAD_WORD>
    # word2index["<EOS_WORD>"] = WEOS_SIGN
    index2word = {word2index[word]: word for word in word2index.keys()}
    dictWord   = (word2index, index2word)
    return dictWord

""" 获取词典 """
def getDataWordDictionary(dataSeqin):
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
    word2index["[UNK]"] = WDUNK_SIGN          # <UNK_WORD>
    word2index["[PAD]"] = WPAD_SIGN          # <PAD_WORD>
    word2index["[EOS]"] = WEOS_SIGN
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
    s = re.sub(r"[^0-9a-zA-Z.!?\-_\' ]+", r"", s)
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


def transIds(pairs, BERTWord2index, DataWord2index, slot2index, intent2index):
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

        itemBERTSeqInIded = [BERTWord2index.get(word, WBUNK_SIGN) for word in itemSeqIn]    # words to ids
        itemDataSeqInIded = [DataWord2index.get(word, WDUNK_SIGN) for word in itemSeqIn]  # words to ids
        itemSeqOutIded = [slot2index.get(slot, SUNK_SIGN) for slot in itemSeqOut]   # slots to ids
        itemIntentIded  = intent2index.get(itemIntent, IUNK_SIGN)                      # labels to ids

        assert len(itemBERTSeqInIded) == len(itemSeqOutIded)
        pairsIded.append([itemBERTSeqInIded, itemDataSeqInIded, itemSeqOutIded, itemIntentIded])

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
        itemIntent = pair[2]

        itemSeqIn  = (itemSeqIn + [WEOS_SIGN] + [WPAD_SIGN] * MAXLEN)[:MAXLEN] if len(itemSeqIn) < MAXLEN else itemSeqIn[:MAXLEN - 1] + [WEOS_SIGN]
        itemSeqOut = (itemSeqOut + [SPAD_SIGN] + [SPAD_SIGN] * MAXLEN)[:MAXLEN] if len(itemSeqIn) < MAXLEN else itemSeqOut[:MAXLEN - 1] + [SPAD_SIGN]

        pairsIdedPaded.append([itemSeqIn, itemSeqOut, itemIntent])

    return pairsIdedPaded


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
                              [item[2] for item in pairs[start:start + batchSize]],
                              [item[3] for item in pairs[start:start + batchSize]]])
    return trainIterator

def getMaxLengthFromBatch(batch, addLength):
    """
    :param batch: 样例对集合
    :return: Batch中最长的输入序列的长度
    """
    return max(MAXLEN, max([len(seqIn) for seqIn in batch[0]])) + addLength

def getSeqInLengthsFromBatch(batch, addLength, MAXLEN=MAXLEN):
    """
    获取计算mask矩阵的有效长度
    :param batch:
    :param addLength:
    :param MAXLEN:
    :return:
    """
    return [min(MAXLEN, len(seqIn) + addLength) for seqIn in batch[0]]


def padBatch(batch, addLength, BERTWord2index, DataWord2index, MAXLEN_TEMP=MAXLEN):
    """
    根据序列最大长度对数据进行裁剪和pading操作
    :param pairsIded: 样例对
    :return:
    """
    batch[0]       = [item[:MAXLEN_TEMP - addLength] for item in batch[0]]
    batch[1]       = [item[:MAXLEN_TEMP - addLength] for item in batch[1]]

    batchBERTSeqIn = [(batch[0][index] + [BERTWord2index["[SEP]"]] + [WPAD_SIGN] * MAXLEN_TEMP)[:MAXLEN_TEMP] for index in range(len(batch[0]))]
    batchDataSeqIn = [(batch[1][index] + [DataWord2index["[EOS]"]] + [WPAD_SIGN] * MAXLEN_TEMP)[:MAXLEN_TEMP] for index in range(len(batch[1]))]
    batchSeqOut    = [(batch[2][index] + [SPAD_SIGN] + [SPAD_SIGN] * MAXLEN_TEMP)[:MAXLEN_TEMP] for index in range(len(batch[2]))]
    for index in range(len(batchBERTSeqIn)): batchBERTSeqIn[index][MAXLEN_TEMP - 1] = BERTWord2index["[SEP]"]
    trainIterator = [batchBERTSeqIn, batchDataSeqIn, batchSeqOut, batch[3]]

    return trainIterator

def vector2Tensor(BatchBERTSeqIn, BatchDataSeqIn, BatchSeqOut, BatchLabel):
    BatchBERTSeqIn  = torch.tensor(BatchBERTSeqIn, dtype=torch.long).cuda() if torch.cuda.is_available() else torch.tensor(BatchBERTSeqIn, dtype=torch.long, device="cpu")
    BatchDataSeqIn = torch.tensor(BatchDataSeqIn, dtype=torch.long).cuda() if torch.cuda.is_available() else torch.tensor(BatchDataSeqIn, dtype=torch.long, device="cpu")
    BatchSeqOut = torch.tensor(BatchSeqOut, dtype=torch.long).cuda() if torch.cuda.is_available() else torch.tensor(BatchSeqOut, dtype=torch.long, device="cpu")
    BatchLabel  = torch.tensor(BatchLabel, dtype=torch.long).cuda() if torch.cuda.is_available() else torch.tensor(BatchLabel, dtype=torch.long, device="cpu")

    return BatchBERTSeqIn, BatchDataSeqIn, BatchSeqOut, BatchLabel
