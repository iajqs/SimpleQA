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
    encoder       = EncoderRNN(input_size=WORDSIZE, emb_size=EMBEDDSIZE, hidden_size=LSTMHIDSIZE, n_layers=NLAYER, dropout=DROPOUT, bidirectional=BIDIRECTIONAL)

    attnIntent    = AttnIntent(hidden_size=LSTMHIDSIZE * MULTI_HIDDEN)
    attnSlot      = AttnSlot(hidden_size=LSTMHIDSIZE * MULTI_HIDDEN)

    decoderIntent = DecoderIntent(hidden_size=LSTMHIDSIZE * MULTI_HIDDEN, intent_size=INTENTSIZE)
    decoderSlot   = DecoderSlot(hidden_size=LSTMHIDSIZE * MULTI_HIDDEN, slot_size=SLOTSIZE)

    seq2Intent    = Seq2Intent(dec_intent=decoderIntent, attn_intent=attnIntent)
    seq2Slots     = Seq2Slots(dec_slot=decoderSlot, attn_slot=attnSlot)

    model         = Seq2Seq(encoder=encoder, seq2Intent=seq2Intent, seq2Slots=seq2Slots)
    #model.apply(init_weights)
    return model


""" 设定模型优化器 """
def initOptimize(model):
    return optim.Adam(model.parameters(), lr=LEARNINGRATE)

""" 设定损失函数 """
def initLossFunction(PAD_IDX=-100):
    return nn.CrossEntropyLoss(ignore_index=PAD_IDX)

""" 训练 """
def train(iter, model=None):

    ''' 读取数据 '''
    dataSeqIn, dataSeqOut, dataLabel = getData(trainDir)                        # 获取原数据
    dictWord        = getWordDictionary(dataSeqIn)                              # 获取词典  (word2index, index2word)
    dictSlot        = getSlotDictionary(dataSeqOut)                             # 获取词槽标签字典  (slot2index, index2slot)
    dictLabel       = getIntentDictionary(dataLabel)                            # 获取意图标签字典  (label2index, index2label)
    pairs           = makePairs(dataSeqIn, dataSeqOut, dataLabel)               # 根据原数据生成样例对    zip(dataSeqIn, dataSeqOut, dataLabel)
    pairsIded       = transIds(pairs, dictWord[0], dictSlot[0], dictLabel[0])   # 将字词都转换为数字id
    pairsIdedPaded  = pad(pairsIded)                                            # 对数据进行pad填充与长度裁剪
    trainIterator   = splitData(pairsIdedPaded)                                 # 讲样例集按BATCHSIZE大小切分成多个块


    ''' 设定字典大小参数 '''
    WORDSIZE   = len(dictWord[0])
    SLOTSIZE   = len(dictSlot[0])
    INTENTSIZE = len(dictLabel[0])

    ''' 定义模型、优化器、损失函数 '''
    model = initModel(WORDSIZE, SLOTSIZE, INTENTSIZE) if model == None else model # 初始化并返回模型

    optimizer = initOptimize(model)                   # 初始化并返回优化器
    criterionLabel = initLossFunction(PAD_IDX)        # 初始化并返回损失函数 -- 意图
    criterionSlot  = initLossFunction()           # 初始化并返回损失函数 -- 词槽

    ''' 模型训练 '''
    model.train()                                   # 设定模型状态为训练状态
    epoch_lossLabel = 0                                  # 定义总损失
    epoch_lossSlot  = 0

    for epoch, batch in enumerate(trainIterator):
        BatchSeqIn  = batch[0]          # 文本序列
        BatchseqOut = batch[1]          # 词槽标签序列
        Batchlabel  = batch[2]          # 意图标签
        BatchSeqIn, BatchseqOut, Batchlabel = vector2Tensor(BatchSeqIn, BatchseqOut, Batchlabel)

        optimizer.zero_grad()

        outputs      = model(BatchSeqIn)
        outputLabel  = outputs[0]
        outputSlots  = outputs[1]

        BatchseqOut  = BatchseqOut.view(BatchseqOut.size(0) * BatchseqOut.size(1))
        outputSlots  = outputSlots.view(outputSlots.size(0) * outputSlots.size(1), SLOTSIZE)

        lossLabel    = criterionLabel(outputLabel, Batchlabel)
        lossSlot     = criterionSlot(outputSlots, BatchseqOut)

        loss = lossLabel + lossSlot
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()

        epoch_lossLabel += lossLabel.item()
        epoch_lossSlot  += lossSlot.item()
        print("iter=%d, epoch=%d / %d: trainLoss = %f、 labelLoss = %f、 slotLoss = %f " % (iter, epoch, len(trainIterator), loss.item(), lossLabel, lossSlot))
    return (epoch_lossLabel / len(trainIterator), epoch_lossSlot / len(trainIterator)),  model, (dictWord, dictSlot, dictLabel)

def evaluate(model, dicts):

    ''' 读取数据 '''
    dataSeqIn, dataSeqOut, dataLabel = getData(validDir)    # 获取原数据
    dictWord  = dicts[0]                                    # 获取词典  (word2index, index2word)
    dictSlot  = dicts[1]                                    # 获取词槽标签字典  (slot2index, index2slot)
    dictLabel = dicts[2]                                    # 获取意图标签字典  (label2index, index2label)
    pairs     = makePairs(dataSeqIn, dataSeqOut, dataLabel)                  # 根据原数据生成样例对    zip(dataSeqIn, dataSeqOut, dataLabel)
    pairsIded = transIds(pairs, dictWord[0], dictSlot[0], dictLabel[0])      # 将字词都转换为数字id
    pairsIdedPaded = pad(pairsIded)                                          # 对数据进行pad填充与长度裁剪
    validIterator  = splitData(pairsIdedPaded)                               # 讲样例集按BATCHSIZE大小切分成多个块

    ''' 设定字典大小参数 '''
    WORDSIZE   = len(dictWord[0])
    SLOTSIZE   = len(dictSlot[0])
    INTENTSIZE = len(dictLabel[0])

    criterionLabel = initLossFunction(PAD_IDX)           # 初始化并返回损失函数 -- 意图
    criterionSlot  = initLossFunction()              # 初始化并返回损失函数 -- 词槽
    ''' 模型验证 '''
    model.eval()
    epoch_lossLabel = 0
    epoch_lossSlot  = 0

    with torch.no_grad():
        for i, batch in enumerate(validIterator):
            BatchSeqIn  = batch[0]
            BatchseqOut = batch[1]
            Batchlabel  = batch[2]
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
    return (epoch_lossLabel / len(validIterator), epoch_lossSlot / len(validIterator))


if __name__ == '__main__':
    model = None
    for iter in range(TRAINITER):
        # print("-" * 20, epoch, "-" * 20)
        trainLoss, model, dicts = train(iter, model)

        # print()

        # time.sleep(100)
        validLoss = evaluate(model, dicts)
        print("iter %d / %d: trainLoss = (label=%f, slot=%f), validLoss = (label=%f, slot=%f)" %
              (iter, TRAINITER, trainLoss[0], trainLoss[1], validLoss[0], validLoss[1]))

    save_model(model, dicts, modelDir + "/base", "base.model", "base.json")

