import sys

if sys.platform == "win32":
    from SimpleQA.Seq2SeqAttention.ouptutUtil import *
    from SimpleQA.Seq2SeqAttention.const import *
    from SimpleQA.Seq2SeqAttention.train import *
    from SimpleQA.Seq2SeqAttention.model import *
else:
    from .Seq2SeqAttention.ouptutUtil import *
    from .Seq2SeqAttention.const import *
    from .Seq2SeqAttention.train import *
    from .Seq2SeqAttention.model import *

import torch

def evaluateLoss(model, dicts, dataDir):
    ''' 读取数据 '''
    dataSeqIn, dataSeqOut, dataLabel = getData(dataDir)    # 获取原数据
    dictWord  = dicts[0]                                    # 获取词典  (word2index, index2word)
    dictSlot  = dicts[1]                                    # 获取词槽标签字典  (slot2index, index2slot)
    dictLabel = dicts[2]                                    # 获取意图标签字典  (label2index, index2label)
    pairs     = makePairs(dataSeqIn, dataSeqOut, dataLabel)                  # 根据原数据生成样例对    zip(dataSeqIn, dataSeqOut, dataLabel)
    pairsIded = transIds(pairs, dictWord[0], dictSlot[0], dictLabel[0])      # 将字词都转换为数字id
    validIterator = splitData(pairsIded, batchSize=64)

    ''' 设定字典大小参数 '''
    WORDSIZE   = len(dictWord[0])
    SLOTSIZE   = len(dictSlot[0])
    INTENTSIZE = len(dictLabel[0])

    criterionLabel = initLossFunction()           # 初始化并返回损失函数 -- 意图
    criterionSlot  = initLossFunction(SPAD_SIGN)  # 初始化并返回损失函数 -- 词槽
    ''' 模型验证 '''
    model.eval()
    epoch_lossLabel = 0
    epoch_lossSlot  = 0

    with torch.no_grad():
        for i, batch in enumerate(validIterator):
            MAXLEN      = getMaxLengthFromBatch(batch, ADDLENGTH)
            lLensSeqin  = getSeqInLengthsFromBatch(batch, ADDLENGTH, 100)
            batch       = padBatch(batch, ADDLENGTH, MAXLEN_TEMP=MAXLEN)  # 按照一个batch一个batch的进行pad
            BatchSeqIn  = batch[0]
            BatchseqOut = batch[1]
            Batchlabel  = batch[2]
            BatchSeqIn, BatchseqOut, Batchlabel = vector2Tensor(BatchSeqIn, BatchseqOut, Batchlabel)

            outputs     = model(BatchSeqIn, lLensSeqin)
            outputLabel = outputs[0]
            outputSlots = outputs[1]


            BatchseqOut = BatchseqOut.view(BatchseqOut.size(0) * BatchseqOut.size(1))
            outputSlots = outputSlots.view(outputSlots.size(0) * outputSlots.size(1), SLOTSIZE)
            lossLabel   = criterionLabel(outputLabel, Batchlabel)
            lossSlot    = criterionSlot(outputSlots, BatchseqOut)

            epoch_lossLabel += lossLabel.item()
            epoch_lossSlot  += lossSlot.item()
    return (epoch_lossLabel / len(validIterator), epoch_lossSlot / len(validIterator))

def evaluateAccuracy(model, dicts, dataDir):
    ''' 读取数据 '''
    dataSeqIn, dataSeqOut, dataLabel = getData(dataDir)     # 获取原数据
    dictWord  = dicts[0]                                    # 获取词典  (word2index, index2word)
    dictSlot  = dicts[1]                                    # 获取词槽标签字典  (slot2index, index2slot)
    dictLabel = dicts[2]                                    # 获取意图标签字典  (label2index, index2label)
    pairs     = makePairs(dataSeqIn, dataSeqOut, dataLabel)                  # 根据原数据生成样例对    zip(dataSeqIn, dataSeqOut, dataLabel)
    pairsIded = transIds(pairs, dictWord[0], dictSlot[0], dictLabel[0])      # 将字词都转换为数字id
    validIterator = splitData(pairsIded, batchSize=64)

    ''' 设定字典大小参数 '''
    WORDSIZE   = len(dictWord[0])
    SLOTSIZE   = len(dictSlot[0])
    INTENTSIZE = len(dictLabel[0])

    ''' 模型验证 '''
    model.eval()
    countLabelAcc = 0
    countSlotAcc  = 0.0

    with torch.no_grad():
        for i, batch in enumerate(validIterator):
            MAXLEN      = getMaxLengthFromBatch(batch, ADDLENGTH)
            lLensSeqin  = getSeqInLengthsFromBatch(batch, ADDLENGTH, 100)
            batch       = padBatch(batch, ADDLENGTH, MAXLEN_TEMP=MAXLEN)  # 按照一个batch一个batch的进行pad
            BatchSeqIn  = batch[0]
            BatchseqOut = batch[1]
            Batchlabel  = batch[2]
            BatchSeqIn, BatchseqOut, Batchlabel = vector2Tensor(BatchSeqIn, BatchseqOut, Batchlabel)

            outputs     = model(BatchSeqIn)
            outputLabel = outputs[0]
            outputSlots = outputs[1]

            """ 正确率计算 """
            _, predictLabel = torch.max(outputLabel, 1)
            _, predictSlot  = torch.max(outputSlots, 2)

            for index in range(len(batch[0])):
                ''' 意图正确率计算 '''
                if batch[2][index] == predictLabel.data.tolist()[index]:
                    countLabelAcc += 1
                ''' 词槽正确率计算 '''
                for index_2 in range(lLensSeqin[index] - 1):
                    # print(index_2, batch[1][index][index_2], predictSlot.data.tolist()[index][index_2])
                    if batch[1][index][index_2] == predictSlot.data.tolist()[index][index_2]:
                        countSlotAcc += 1 / (lLensSeqin[index] - 1)

    return (countLabelAcc / len(pairsIded), countSlotAcc / len(pairsIded))


def test_predict(model):

    batch = []
    batch.append([[1, 2, 3, 4], [1, 2, 3, 4]])
    batch.append([[1, 2, 3, 4]])
    batch.append([1, 2])

    lLensSeqin = getSeqInLengthsFromBatch(batch, ADDLENGTH, MAXLEN)
    batch = padBatch(batch, MAXLEN_TEMP=MAXLEN)
    BatchSeqIn = batch[0]
    BatchseqOut = batch[1]
    Batchlabel = batch[2]
    BatchSeqIn, BatchseqOut, Batchlabel = vector2Tensor(BatchSeqIn, BatchseqOut, Batchlabel)

    print(BatchSeqIn, lLensSeqin)
    model.eval()
    outputs = model.encoder.embedding(BatchSeqIn)
    outputs, (hidden, cell) = model.encoder.lstm(outputs)
    print(outputs.size())
    print(outputs[0][0])
    print(outputs[1][0])

modelLoaded, dicts = load_model(modelDir + "/base", "base.model", "base.json")

WORDSIZE    = len(dicts[0][0])
SLOTSIZE    = len(dicts[1][0])
INTENTSIZE  = len(dicts[2][0])

model = initModel(WORDSIZE, SLOTSIZE, INTENTSIZE, isTrain=False)
model.load_state_dict(modelLoaded)
model.eval()

# test_predict(model)

print(WORDSIZE, SLOTSIZE, INTENTSIZE)

print(evaluateLoss(model, dicts, testDir))
print(evaluateAccuracy(model, dicts, testDir))


# (0.5918484792594478, 0.35097516863997963)
# (0.8443449048152296, 0.9170913960089957)