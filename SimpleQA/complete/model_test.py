import sys
import numpy as np

if sys.platform == "win32":
    from SimpleQA.complete.ouptutUtil import *
    from SimpleQA.complete.const import *
    from SimpleQA.complete.train import *
    from SimpleQA.complete.model import *
    from SimpleQA.complete.utils import *
else:
    from .ouptutUtil import *
    from .const import *
    from .train import *
    from .model import *
    from .utils import *

import torch

def processBatch(batch, dictBERTWord, dictDataWord):
    MAXLEN = getMaxLengthFromBatch(batch, ADDLENGTH)
    lLensSeqin = getSeqInLengthsFromBatch(batch, ADDLENGTH, MAXLEN)
    batch = padBatch(batch, ADDLENGTH, BERTWord2index=dictBERTWord[0], DataWord2index=dictDataWord[0], MAXLEN_TEMP=MAXLEN)  # 按照一个batch一个batch的进行pad

    BatchBERTSeqIn = batch[0]
    BatchDataSeqIn = batch[1]
    BatchSeqOut    = batch[2]
    BatchLabel     = batch[3]
    BatchDomain    = batch[4]
    BatchBERTSeqIn, BatchDataSeqIn, BatchSeqOut, BatchLabel, BatchDomain = vector2Tensor(BatchBERTSeqIn, BatchDataSeqIn, BatchSeqOut, BatchLabel, BatchDomain)

    return MAXLEN, lLensSeqin, BatchBERTSeqIn, BatchDataSeqIn, BatchSeqOut, BatchLabel, BatchDomain


def evaluateLoss(model, dicts, dataDir):
    dataSeqIn, dataSeqOut, dataIntent, dataDomain = getData(validDir)     # 获取原数据
    dictBERTWord = dicts[0]                                     # 获取词典  (word2index, index2word)
    dictDataWord = dicts[1]
    dictSlot     = dicts[2]                                     # 获取词槽标签字典  (slot2index, index2slot)
    dictIntent   = dicts[3]                                     # 获取意图标签字典  (label2index, index2label)
    dictDomain   = dicts[4]
    pairs      = makePairs(dataSeqIn, dataSeqOut, dataIntent, dataDomain)                   # 根据原数据生成样例对    zip(dataSeqIn, dataSeqOut, dataIntent)
    pairsIded  = transIds(pairs, dictBERTWord[0], dictDataWord[0], dictSlot[0], dictIntent[0], dictDomain[0])       # 将字词都转换为数字id

    validIterator  = splitData(pairsIded)                                       # 讲样例集按BATCHSIZE大小切分成多个块

    ''' 设定字典大小参数 '''
    WORDSIZE   = len(dictDataWord[0])
    SLOTSIZE   = len(dictSlot[0])
    INTENTSIZE = len(dictIntent[0])

    criterionDomain = initLossFunction()          # 初始化并返回损失函数 -- 领域
    criterionLabel  = initLossFunction()           # 初始化并返回损失函数 -- 意图
    criterionSlot   = initLossFunction(SPAD_SIGN)  # 初始化并返回损失函数 -- 词槽
    ''' 模型验证 '''
    model.eval()
    epoch_lossDomain = 0
    epoch_lossLabel  = 0
    epoch_lossSlot   = 0

    with torch.no_grad():
        for i, batch in enumerate(validIterator):
            MAXLEN, lLensSeqin, BatchBERTSeqIn, BatchDataSeqIn, BatchSeqOut, BatchLabel, BatchDomain\
                = processBatch(batch, dictBERTWord, dictDataWord)

            outputs, slot_crf = model(BatchBERTSeqIn, BatchDataSeqIn, BatchSeqOut, lLensSeqin)
            outputDomain = outputs[0]
            outputLabel  = outputs[1]
            outputSlots  = outputs[2]

            BatchSeqOut = BatchSeqOut.view(BatchSeqOut.size(0) * BatchSeqOut.size(1))
            outputSlots = outputSlots.view(outputSlots.size(0) * outputSlots.size(1), SLOTSIZE)

            lossDomain  = criterionDomain(outputDomain, BatchDomain)
            lossLabel   = criterionLabel(outputLabel, BatchLabel)
            lossSlot    = criterionSlot(outputSlots, BatchSeqOut)

            epoch_lossDomain += lossDomain.item()
            epoch_lossLabel += lossLabel.item()
            epoch_lossSlot  += lossSlot.item()
    return (epoch_lossLabel / len(validIterator), epoch_lossSlot / len(validIterator))

def evaluateAccuracy(model, dicts, dataDir):
    ''' 读取数据 '''
    dataSeqIn, dataSeqOut, dataIntent, dataDomain = getData(validDir)     # 获取原数据
    dictBERTWord = dicts[0]                                     # 获取词典  (word2index, index2word)
    dictDataWord = dicts[1]
    dictSlot     = dicts[2]                                     # 获取词槽标签字典  (slot2index, index2slot)
    dictIntent   = dicts[3]                                     # 获取意图标签字典  (label2index, index2label)
    dictDomain   = dicts[4]
    pairs        = makePairs(dataSeqIn, dataSeqOut, dataIntent, dataDomain)                   # 根据原数据生成样例对    zip(dataSeqIn, dataSeqOut, dataIntent)
    pairsIded    = transIds(pairs, dictBERTWord[0], dictDataWord[0], dictSlot[0], dictIntent[0], dictDomain[0])       # 将字词都转换为数字id
    validIterator  = splitData(pairsIded)                                       # 讲样例集按BATCHSIZE大小切分成多个块


    ''' 设定字典大小参数 '''
    WORDSIZE   = len(dictDataWord[0])
    SLOTSIZE   = len(dictSlot[0])
    INTENTSIZE = len(dictIntent[0])

    ''' 模型验证 '''
    model.eval()

    correct_domains = []
    pred_domains    = []
    correct_intents = []
    pred_intents    = []
    correct_slots   = []
    pred_slots      = []

    # count = 0
    with torch.no_grad():
        for i, batch in enumerate(validIterator):
            MAXLEN, lLensSeqin, BatchBERTSeqIn, BatchDataSeqIn, BatchSeqOut, BatchLabel, BatchDomain\
                = processBatch(batch, dictBERTWord, dictDataWord)
            outputs, slot_crf = model(BatchBERTSeqIn, BatchDataSeqIn, BatchSeqOut, lLensSeqin)
            outputDomain = outputs[0]
            outputLabel  = outputs[1]
            outputSlots  = outputs[2]

            """ 正确率计算 """
            _, predictDomain = torch.max(outputDomain, 1)
            _, predictLabel  = torch.max(outputLabel, 1)
            _, predictSlot   = torch.max(outputSlots, 2)
            predictSlot = model.seq2Slots.crf.CRF.decode(outputSlots)

            for index in range(len(batch[0])):
                correct_domains.append(batch[4][index])
                correct_intents.append(batch[3][index])
                correct_slots.append([dictSlot[1][str(item)] for item in batch[2][index][:lLensSeqin[index] - 1]])

                pred_domains.append(predictDomain.data.tolist()[index])
                pred_intents.append(predictLabel.data.tolist()[index])
                pred_slots.append([dictSlot[1][str(item)] for item in predictSlot[index][:lLensSeqin[index] - 1]])

    correct_intents = np.array(correct_intents)
    correct_domains = np.array(correct_domains)
    pred_intents    = np.array(pred_intents)
    pred_domains    = np.array(pred_domains)
    accuracyIntent  = (correct_intents==pred_intents)
    semantic_error  = computeSentence(accuracyIntent, correct_slots, pred_slots)
    accuracyIntent  = np.mean(accuracyIntent.astype(float)) * 100.0
    accuracyDomain  = (correct_domains==pred_domains)
    accuracyDomain  = np.mean(accuracyDomain.astype(float)) * 100.0
    f1, precision, recall = computeF1Score(correct_slots, pred_slots)
    return ("acc domain", accuracyDomain, "acc_intent:", accuracyIntent, "slot, precision=%f, recall=%f, f1=%f" % (precision, recall, f1), "semantic error(intent, slots are all correct): %f", semantic_error)



def test():
    modelLoaded, dicts = load_model(modelDir + "/complete", "complete.model", "complete.json")

    WORDSIZE    = len(dicts[1][0])
    SLOTSIZE    = len(dicts[2][0])
    INTENTSIZE  = len(dicts[3][0])
    DOMAINSIZE  = len(dicts[4][0])
    model = initModel(WORDSIZE, SLOTSIZE, INTENTSIZE, DOMAINSIZE, isTrain=False)
    model.load_state_dict(modelLoaded)
    model.eval()
    # print(WORDSIZE, SLOTSIZE, INTENTSIZE)

    print(evaluateLoss(model, dicts, testDir))      # (0.302971917404128, 0.1861887446471623)
    print(evaluateAccuracy(model, dicts, testDir))  # (0.9540873460246361, 0.966896114981263)

if __name__ == '__main__':
    test()