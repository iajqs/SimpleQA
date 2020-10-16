import sys
print(sys.platform)
if sys.platform == "win32":
    from SimpleQA.Seq2SeqAttention.model import *
    from SimpleQA.Seq2SeqAttention.const import *
    from SimpleQA.Seq2SeqAttention.dataUtil import *
    from SimpleQA.Seq2SeqAttention.ouptutUtil import *
else:
    from .model import *
    from .const import *
    from .dataUtil import *
    from .ouptutUtil import *

import torch
import torch.nn as nn
import torch.optim as optim

import tqdm

""" 初始化模型 """
def init_weights(model):
    for param in model.parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

def initModel(WORDSIZE, SLOTSIZE, INTENTSIZE, isTrain=True):
    """
    初始化所有模型原件，并将原件传给总模型框架
    EncoderRNN、AttnIntent、DecoderIntent -> Seq2Intent
    EncoderRNN、AttnSlot、DecoderSlot -> Seq2Slot
    Seq2Intent、Seq2Slot -> Seq2Seq
    :return:
    """
    encoder       = EncoderRNN(input_size=WORDSIZE, emb_size=EMBEDDSIZE, pading_idx=WPAD_SIGN, hidden_size=LSTMHIDSIZE, n_layers=NLAYER, dropout=DROPOUT, bidirectional=BIDIRECTIONAL)

    attnIntent    = AttnIntent()
    attnSlot      = AttnSlot()

    decoderIntent = DecoderIntent(hidden_size=LSTMHIDSIZE * MULTI_HIDDEN, intent_size=INTENTSIZE)
    decoderSlot   = DecoderSlot(hidden_size=LSTMHIDSIZE * MULTI_HIDDEN, slot_size=SLOTSIZE)

    seq2Intent    = Seq2Intent(dec_intent=decoderIntent, attn_intent=attnIntent)
    seq2Slots     = Seq2Slots(dec_slot=decoderSlot, attn_slot=attnSlot, hidden_size=LSTMHIDSIZE * MULTI_HIDDEN)

    model         = Seq2Seq(encoder=encoder, seq2Intent=seq2Intent, seq2Slots=seq2Slots)

    if isTrain:
        model.apply(init_weights)
    return model


""" 设定模型优化器 """
def initOptimize(model):
    return optim.Adam(model.parameters(), lr=LEARNINGRATE)

""" 设定损失函数 """
def initLossFunction(PAD_IDX=-100):
    return nn.CrossEntropyLoss(ignore_index=PAD_IDX)

""" 训练 """
def train(iter, model=None, optimizer=None, isTrainSlot=True, isTrainIntent=True):
    ''' 读取数据 '''
    dataSeqIn, dataSeqOut, dataIntent = getData(trainDir)                       # 获取原数据
    dictWord        = getWordDictionary(dataSeqIn)                              # 获取词典  (word2index, index2word)
    dictSlot        = getSlotDictionary(dataSeqOut)                             # 获取词槽标签字典  (slot2index, index2slot)
    dictIntent      = getIntentDictionary(dataIntent)                           # 获取意图标签字典  (intent2index, index2intent)
    pairs           = makePairs(dataSeqIn, dataSeqOut, dataIntent)              # 根据原数据生成样例对    zip(dataSeqIn, dataSeqOut, dataIntent)
    pairsIded       = transIds(pairs, dictWord[0], dictSlot[0], dictIntent[0])  # 将字词都转换为数字id
    # pairsIdedPaded  = pad(pairsIded)                                          # 对数据进行pad填充与长度裁剪
    trainIterator   = splitData(pairsIded)                                      # 讲样例集按BATCHSIZE大小切分成多个块


    ''' 设定字典大小参数 '''
    WORDSIZE   = len(dictWord[0])
    SLOTSIZE   = len(dictSlot[0])
    INTENTSIZE = len(dictIntent[0])

    ''' 定义模型、优化器、损失函数 '''
    model = initModel(WORDSIZE, SLOTSIZE, INTENTSIZE) if model == None else model # 初始化并返回模型

    optimizer       = initOptimize(model) if optimizer == None else optimizer              # 初始化并返回优化器
    criterionIntent = initLossFunction()                 # 初始化并返回损失函数 -- 意图
    criterionSlot   = initLossFunction(SPAD_SIGN)        # 初始化并返回损失函数 -- 词槽

    ''' 模型训练 '''
    model.train()                                       # 设定模型状态为训练状态
    epoch_lossIntent = 0                                # 定义总损失
    epoch_lossSlot   = 0

    for epoch, batch in tqdm.tqdm(enumerate(trainIterator)):
        MAXLEN      = getMaxLengthFromBatch(batch, ADDLENGTH)
        lLensSeqin  = getSeqInLengthsFromBatch(batch, ADDLENGTH, MAXLEN=MAXLEN)
        batch       = padBatch(batch, ADDLENGTH, MAXLEN_TEMP=MAXLEN)   # 按照一个batch一个batch的进行pad
        BatchSeqIn  = batch[0]          # 文本序列
        BatchSeqOut = batch[1]          # 词槽标签序列
        BatchIntent = batch[2]          # 意图标签
        BatchSeqIn, BatchSeqOut, BatchIntent = vector2Tensor(BatchSeqIn, BatchSeqOut, BatchIntent)

        optimizer.zero_grad()

        outputs      = model(BatchSeqIn, lLensSeqin)
        outputIntent = outputs[0]
        outputSlots  = outputs[1]

        BatchSeqOut  = BatchSeqOut.view(BatchSeqOut.size(0) * BatchSeqOut.size(1))
        outputSlots  = outputSlots.view(outputSlots.size(0) * outputSlots.size(1), SLOTSIZE)

        lossIntent   = criterionIntent(outputIntent, BatchIntent)
        lossSlot     = criterionSlot(outputSlots, BatchSeqOut)

        loss = lossIntent * 0
        loss = loss + lossIntent if isTrainIntent == True else loss
        loss = loss + lossSlot if isTrainSlot == True else loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()

        epoch_lossIntent += lossIntent.item()
        epoch_lossSlot  += lossSlot.item()

        import time
        time.sleep(0.4)
        # print("iter=%d, epoch=%d / %d: MAXLEN = %d; trainLoss = %f、 intentLoss = %f、 slotLoss = %f " % (iter, epoch, len(trainIterator), MAXLEN, loss.item(), lossIntent, lossSlot))

    return (epoch_lossIntent / len(trainIterator), epoch_lossSlot / len(trainIterator)),  model, optimizer, (dictWord, dictSlot, dictIntent)

def evaluate(model, dicts):

    ''' 读取数据 '''
    dataSeqIn, dataSeqOut, dataIntent = getData(validDir)     # 获取原数据
    dictWord   = dicts[0]                                     # 获取词典  (word2index, index2word)
    dictSlot   = dicts[1]                                     # 获取词槽标签字典  (slot2index, index2slot)
    dictIntent = dicts[2]                                     # 获取意图标签字典  (label2index, index2label)
    pairs      = makePairs(dataSeqIn, dataSeqOut, dataIntent)                   # 根据原数据生成样例对    zip(dataSeqIn, dataSeqOut, dataIntent)
    pairsIded  = transIds(pairs, dictWord[0], dictSlot[0], dictIntent[0])       # 将字词都转换为数字id

    validIterator  = splitData(pairsIded)                                       # 讲样例集按BATCHSIZE大小切分成多个块

    ''' 设定字典大小参数 '''
    WORDSIZE   = len(dictWord[0])
    SLOTSIZE   = len(dictSlot[0])
    INTENTSIZE = len(dictIntent[0])

    criterionIntent = initLossFunction()              # 初始化并返回损失函数 -- 意图
    criterionSlot   = initLossFunction(SPAD_SIGN)     # 初始化并返回损失函数 -- 词槽
    ''' 模型验证 '''
    model.eval()
    epoch_lossIntent = 0
    epoch_lossSlot  = 0

    with torch.no_grad():
        for i, batch in enumerate(validIterator):
            MAXLEN      = getMaxLengthFromBatch(batch, ADDLENGTH)
            lLensSeqin  = getSeqInLengthsFromBatch(batch, ADDLENGTH, MAXLEN=MAXLEN)
            batch       = padBatch(batch, ADDLENGTH, MAXLEN_TEMP=MAXLEN)  # 按照一个batch一个batch的进行pad
            BatchSeqIn  = batch[0]  # 文本序列
            BatchSeqOut = batch[1]  # 词槽标签序列
            BatchIntent = batch[2]  # 意图标签
            BatchSeqIn, BatchSeqOut, BatchIntent = vector2Tensor(BatchSeqIn, BatchSeqOut, BatchIntent)

            outputs      = model(BatchSeqIn, lLensSeqin)
            outputIntent = outputs[0]
            outputSlots  = outputs[1]

            BatchSeqOut = BatchSeqOut.view(BatchSeqOut.size(0) * BatchSeqOut.size(1))
            outputSlots = outputSlots.view(outputSlots.size(0) * outputSlots.size(1), SLOTSIZE)

            lossIntent   = criterionIntent(outputIntent, BatchIntent)
            lossSlot     = criterionSlot(outputSlots, BatchSeqOut)

            epoch_lossIntent += lossIntent.item()
            epoch_lossSlot   += lossSlot.item()
    return (epoch_lossIntent / len(validIterator), epoch_lossSlot / len(validIterator))


if __name__ == '__main__':
    modelBest = None
    model     = None
    optimizer = None
    lossMin   = 100

    for iter in range(TRAINITER):
        trainLoss, model, optimizer, dicts = train(iter, model=model, optimizer=optimizer, isTrainIntent=True, isTrainSlot=True)

        validLoss = evaluate(model, dicts)
        print("iter %d / %d: trainLoss = (intent=%f, slot=%f), validLoss = (intent=%f, slot=%f)" %
              (iter, TRAINITER, trainLoss[0], trainLoss[1], validLoss[0], validLoss[1]))
        if validLoss[0] + validLoss[1] < lossMin:
            lossMin = validLoss[0] + validLoss[1]
            modelBest = model
            save_model(modelBest, dicts, modelDir + "/base", "base.model", "base.json")
        if trainLoss[0] + trainLoss[1] < 0.1:
            for p in optimizer.param_groups:
                p['lr'] *= 0.9
        elif trainLoss[0] + trainLoss[1] < 0.2:
            LEARNINGRATE = 1e-3
            for p in optimizer.param_groups:
                p['lr'] = 1e-3


    # save_model(modelBest, dicts, modelDir + "/base", "base.model", "base.json")

