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

def evaluateLoss(model, dicts):
    ''' 读取数据 '''
    dataSeqIn, dataSeqOut, dataLabel = getData(validDir)    # 获取原数据
    dictWord  = dicts[0]                                    # 获取词典  (word2index, index2word)
    dictSlot  = dicts[1]                                    # 获取词槽标签字典  (slot2index, index2slot)
    dictLabel = dicts[2]                                    # 获取意图标签字典  (label2index, index2label)
    pairs     = makePairs(dataSeqIn, dataSeqOut, dataLabel)                  # 根据原数据生成样例对    zip(dataSeqIn, dataSeqOut, dataLabel)
    pairsIded = transIds(pairs, dictWord[0], dictSlot[0], dictLabel[0])      # 将字词都转换为数字id

    ''' 设定字典大小参数 '''
    WORDSIZE   = len(dictWord[0])
    SLOTSIZE   = len(dictSlot[0])
    INTENTSIZE = len(dictLabel[0])

    criterionLabel = initLossFunction(PAD_IDX)           # 初始化并返回损失函数 -- 意图
    criterionSlot  = initLossFunction()                  # 初始化并返回损失函数 -- 词槽
    ''' 模型验证 '''
    model.eval()
    epoch_lossLabel = 0
    epoch_lossSlot  = 0

    with torch.no_grad():
        for i, batch in enumerate(pairsIded):
            BatchSeqIn  = [batch[0]]
            BatchseqOut = [batch[1]]
            Batchlabel  = [batch[2]]
            BatchSeqIn, BatchseqOut, Batchlabel = vector2Tensor(BatchSeqIn, BatchseqOut, Batchlabel)

            outputs     = model(BatchSeqIn)
            outputLabel = outputs[0]
            outputSlots = outputs[1]

            BatchseqOut = BatchseqOut.view(BatchseqOut.size(0) * BatchseqOut.size(1))
            outputSlots = outputSlots.view(outputSlots.size(0) * outputSlots.size(1), SLOTSIZE)

            lossLabel   = criterionLabel(outputLabel, Batchlabel)
            lossSlot    = criterionSlot(outputSlots, BatchseqOut)

            epoch_lossLabel += lossLabel.item()
            epoch_lossSlot  += lossSlot.item()
    return (epoch_lossLabel / len(pairsIded), epoch_lossSlot / len(pairsIded))

def evaluateAccuracy(model, dicts):

    ''' 读取数据 '''
    dataSeqIn, dataSeqOut, dataLabel = getData(validDir)    # 获取原数据
    dictWord  = dicts[0]                                    # 获取词典  (word2index, index2word)
    dictSlot  = dicts[1]                                    # 获取词槽标签字典  (slot2index, index2slot)
    dictLabel = dicts[2]                                    # 获取意图标签字典  (label2index, index2label)
    pairs     = makePairs(dataSeqIn, dataSeqOut, dataLabel)                  # 根据原数据生成样例对    zip(dataSeqIn, dataSeqOut, dataLabel)
    pairsIded = transIds(pairs, dictWord[0], dictSlot[0], dictLabel[0])      # 将字词都转换为数字id

    ''' 设定字典大小参数 '''
    WORDSIZE   = len(dictWord[0])
    SLOTSIZE   = len(dictSlot[0])
    INTENTSIZE = len(dictLabel[0])


    ''' 模型验证 '''
    model.eval()
    countLabelAcc = 0
    countSlotAcc  = 0.0

    with torch.no_grad():
        for i, batch in enumerate(pairsIded):
            BatchSeqIn  = [batch[0]]
            BatchseqOut = [batch[1]]
            Batchlabel  = [batch[2]]
            BatchSeqIn, BatchseqOut, Batchlabel = vector2Tensor(BatchSeqIn, BatchseqOut, Batchlabel)

            outputs     = model(BatchSeqIn)
            outputLabel = outputs[0]
            outputSlots = outputs[1].squeeze(0)

            """ 正确率计算 """
            _, predictLabel = torch.max(outputLabel, 1)
            _, predictSlot  = torch.max(outputSlots, 1)

            # print(predictSlot)
            ''' 意图正确率计算 '''
            if batch[2] == predictLabel.data.tolist()[0]:
                countLabelAcc += 1
            ''' 词槽正确率计算 '''
            for i in range(len(batch[1])):
                if batch[1][i] == predictSlot.data.tolist()[i]:
                    countSlotAcc += 1 / len(batch[1])

    return (countLabelAcc / len(pairsIded), countSlotAcc / len(pairsIded))


modelLoaded, dicts = load_model(modelDir + "/base", "base.model", "base.json")

WORDSIZE    = len(dicts[0][0])
SLOTSIZE    = len(dicts[1][0])
INTENTSIZE  = len(dicts[2][0])

model = initModel(WORDSIZE, SLOTSIZE, INTENTSIZE)
model.load_state_dict(modelLoaded)
model.eval()

print(evaluateLoss(model, dicts))
print(evaluateAccuracy(model, dicts))