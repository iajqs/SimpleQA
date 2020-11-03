import sys
print(sys.platform)
if sys.platform == "win32":
    from SimpleQA.complete.model import *
    from SimpleQA.complete.model_test import *
    from SimpleQA.complete.const import *
    from SimpleQA.complete.dataUtil import *
    from SimpleQA.complete.ouptutUtil import *
else:
    from .model import *
    from .model_test import *
    from .const import *
    from .dataUtil import *
    from .ouptutUtil import *

import torch
import torch.nn as nn
import torch.optim as optim

from pytorch_pretrained_bert import BertTokenizer, BertModel, BertAdam
import tqdm

""" 初始化模型 """
def init_weights(model):
    for param in model.parameters():
        nn.init.uniform_(param.data, -0.1, 0.1)

def initModel(WORDSIZE, SLOTSIZE, INTENTSIZE, DOMAINSIZE, isTrain=True):
    """
    初始化所有模型原件，并将原件传给总模型框架
    EncoderRNN、AttnIntent、DecoderIntent -> Seq2Intent
    EncoderRNN、AttnSlot、DecoderSlot -> Seq2Slot
    Seq2Intent、Seq2Slot -> Seq2Seq
    :return:
    """
    BERTModel     = BertModel.from_pretrained(modelDir + "\\bert\\bert-base-uncased")

    encoder       = EncoderRNN(BERTModel, input_size=WORDSIZE, emb_size=BERTSIZE // MULTI_HIDDEN, pading_idx=WPAD_SIGN,
                         hidden_size=BERTSIZE // MULTI_HIDDEN, n_layers=NLAYER, dropout=DROPOUT, bidirectional=BIDIRECTIONAL)

    attnDomain    = AttnDomain()
    attnIntent    = AttnIntent()
    attnSlot      = AttnSlot()

    decoderDomain = DecoderDomain(hidden_size=ENCODERSIZE, domain_size=DOMAINSIZE)
    decoderIntent = DecoderIntent(hidden_size=INTENTHIDSIZE, intent_size=INTENTSIZE)
    decoderSlot   = DecoderSlot(hidden_size=ENCODERSIZE, slot_size=SLOTSIZE)

    crf           = SlotCRF(num_tags=SLOTSIZE)

    seq2Domain    = Seq2Domain(dec_domain=decoderDomain, attn_domain=attnDomain)
    seq2Intent    = Seq2Intent(dec_intent=decoderIntent, attn_intent=attnIntent, EncoderHidSize=ENCODERSIZE, IntentHidSize=INTENTHIDSIZE)
    seq2Slots     = Seq2Slots(dec_slot=decoderSlot, attn_slot=attnSlot, crf=crf, IntentHidSize=INTENTHIDSIZE, hidden_size=ENCODERSIZE)

    model         = Seq2Seq(encoder=encoder, seq2Intent=seq2Intent, seq2Slots=seq2Slots, seq2Domain=seq2Domain)
    model         = model.cuda() if torch.cuda.is_available() else model

    # if isTrain:
    #     model.apply(init_weights)

    return model


""" 设定模型优化器 """
def initOptimize(model):
    return optim.Adam(model.parameters(), lr=LEARNINGRATE)

""" 设定损失函数 """
def initLossFunction(PAD_IDX=-100):
    return nn.CrossEntropyLoss(ignore_index=PAD_IDX)

""" 训练 """
def train(iter, model=None, optimizer=None, isTrainSlot=True, isTrainIntent=True, isTrainDomain=True, ratioIntent = 1, ratioSlot = 1):
    ''' 读取数据 '''
    dataSeqIn, dataSeqOut, dataIntent, dataDomain = getData(trainDir)                       # 获取原数据
    tokenizer       = BertTokenizer.from_pretrained(modelDir + "\\bert\\uncased") #
    dictBERTWord    = getBERTWordDictionary(tokenizer)                          # 获取词典  (word2index, index2word)
    dictDataWord    = getDataWordDictionary(dataSeqIn)                          # 获取词典
    dictSlot        = getSlotDictionary(dataSeqOut)                             # 获取词槽标签字典  (slot2index, index2slot)
    dictIntent      = getIntentDictionary(dataIntent)                           # 获取意图标签字典  (intent2index, index2intent)
    dictDomain      = getDomainDictionary(dataDomain)
    pairs           = makePairs(dataSeqIn, dataSeqOut, dataIntent, dataDomain)              # 根据原数据生成样例对    zip(dataSeqIn, dataSeqOut, dataIntent)
    pairsIded       = transIds(pairs, dictBERTWord[0], dictDataWord[0], dictSlot[0], dictIntent[0], dictDomain[0])  # 将字词都转换为数字id
    # pairsIdedPaded  = pad(pairsIded)                                          # 对数据进行pad填充与长度裁剪
    trainIterator   = splitData(pairsIded)                                      # 讲样例集按BATCHSIZE大小切分成多个块
    trainIterator = trainIterator[:len(trainIterator) - len(trainIterator) // 10]

    ''' 设定字典大小参数 '''
    WORDSIZE   = len(dictDataWord[0])
    SLOTSIZE   = len(dictSlot[0])
    INTENTSIZE = len(dictIntent[0])
    DOMAINSIZE = len(dictDomain[0])
    ''' 定义模型、优化器、损失函数 '''
    model = initModel(WORDSIZE, SLOTSIZE, INTENTSIZE, DOMAINSIZE) if model == None else model # 初始化并返回模型

    optimizer       = initOptimize(model) if optimizer == None else optimizer              # 初始化并返回优化器
    criterionDomain = initLossFunction()                 # 初始化并返回损失函数 -- 领域
    criterionIntent = initLossFunction()                 # 初始化并返回损失函数 -- 意图
    criterionSlot   = initLossFunction(SPAD_SIGN)        # 初始化并返回损失函数 -- 词槽

    ''' 模型训练 '''
    model.train()                                       # 设定模型状态为训练状态
    epoch_lossDomain = 0
    epoch_lossIntent = 0                                # 定义总损失
    epoch_lossSlot   = 0
    epoch_lossCRF    = 0
    for epoch, batch in tqdm.tqdm(enumerate(trainIterator)):
        MAXLEN      = getMaxLengthFromBatch(batch, ADDLENGTH)
        lLensSeqin  = getSeqInLengthsFromBatch(batch, ADDLENGTH, MAXLEN=MAXLEN)
        batch       = padBatch(batch, ADDLENGTH, BERTWord2index=dictBERTWord[0], DataWord2index=dictDataWord[0], MAXLEN_TEMP=MAXLEN)   # 按照一个batch一个batch的进行pad
        BatchBERTSeqIn  = batch[0]          # BERT文本序列
        BatchDataSeqIn  = batch[1]          # Data文本序列
        BatchSeqOut = batch[2]          # 词槽标签序列
        BatchIntent = batch[3]          # 意图标签
        BatchDomain = batch[4]
        BatchBERTSeqIn, BatchDataSeqIn, BatchSeqOut, BatchIntent, BatchDomain = vector2Tensor(BatchBERTSeqIn, BatchDataSeqIn, BatchSeqOut, BatchIntent, BatchDomain)
        optimizer.zero_grad()

        outputs, slot_crf = model(BatchBERTSeqIn, BatchDataSeqIn, BatchSeqOut, lLensSeqin)

        outputDomain = outputs[0]
        outputIntent = outputs[1]
        outputSlots  = outputs[2]

        BatchSeqOut  = BatchSeqOut.view(BatchSeqOut.size(0) * BatchSeqOut.size(1))
        outputSlots  = outputSlots.view(outputSlots.size(0) * outputSlots.size(1), SLOTSIZE)

        lossDomain   = criterionDomain(outputDomain, BatchDomain)
        lossIntent   = criterionIntent(outputIntent, BatchIntent)
        lossSlot     = criterionSlot(outputSlots, BatchSeqOut)

        loss = lossIntent * 0
        loss = loss + lossDomain if isTrainDomain == True else loss
        loss = loss + lossIntent if isTrainIntent == True else loss
        loss = loss + lossSlot if isTrainSlot == True else loss
        loss = loss - slot_crf
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()

        epoch_lossDomain += lossDomain.item()
        epoch_lossIntent += lossIntent.item()
        epoch_lossSlot   += lossSlot.item()
        epoch_lossCRF    -= slot_crf.item()

    return (epoch_lossDomain / len(trainIterator), epoch_lossIntent / len(trainIterator), epoch_lossSlot / len(trainIterator), epoch_lossCRF / len(trainIterator)),  model, optimizer, (dictBERTWord, dictDataWord, dictSlot, dictIntent, dictDomain)

def evaluate(model, dicts):

    ''' 读取数据 '''
    dataSeqIn, dataSeqOut, dataIntent, dataDomain = getData(validDir)     # 获取原数据
    dictBERTWord = dicts[0]                                     # 获取词典  (word2index, index2word)
    dictDataWord = dicts[1]
    dictSlot     = dicts[2]                                     # 获取词槽标签字典  (slot2index, index2slot)
    dictIntent   = dicts[3]                                     # 获取意图标签字典  (label2index, index2label)
    dictDomain   = dicts[4]
    pairs      = makePairs(dataSeqIn, dataSeqOut, dataIntent, dataDomain)                   # 根据原数据生成样例对    zip(dataSeqIn, dataSeqOut, dataIntent)
    pairsIded  = transIds(pairs, dictBERTWord[0], dictDataWord[0], dictSlot[0], dictIntent[0], dictDomain[0])       # 将字词都转换为数字id

    validIterator  = splitData(pairsIded)                                       # 讲样例集按BATCHSIZE大小切分成多个块
                                  # 讲样例集按BATCHSIZE大小切分成多个块


    ''' 设定字典大小参数 '''
    WORDSIZE   = len(dictDataWord[0])
    SLOTSIZE   = len(dictSlot[0])
    INTENTSIZE = len(dictIntent[0])
    DOMAINSIZE = len(dictDomain[0])

    criterionDomain = initLossFunction()              # 初始化并返回损失函数 -- 领域
    criterionIntent = initLossFunction()              # 初始化并返回损失函数 -- 意图
    criterionSlot   = initLossFunction(SPAD_SIGN)     # 初始化并返回损失函数 -- 词槽
    ''' 模型验证 '''
    model.eval()
    epoch_lossDomain = 0
    epoch_lossIntent = 0
    epoch_lossSlot   = 0
    epoch_lossCRF    = 0
    with torch.no_grad():
        for i, batch in enumerate(validIterator):
            MAXLEN = getMaxLengthFromBatch(batch, ADDLENGTH)
            lLensSeqin = getSeqInLengthsFromBatch(batch, ADDLENGTH, MAXLEN=MAXLEN)
            batch = padBatch(batch, ADDLENGTH, BERTWord2index=dictBERTWord[0], DataWord2index=dictDataWord[0],
                             MAXLEN_TEMP=MAXLEN)  # 按照一个batch一个batch的进行pad
            BatchBERTSeqIn = batch[0]  # BERT文本序列
            BatchDataSeqIn = batch[1]  # Data文本序列
            BatchSeqOut = batch[2]  # 词槽标签序列
            BatchIntent = batch[3]  # 意图标签
            BatchDomain = batch[4]  # 领域标签
            # print(BatchSeqIn[0])
            BatchBERTSeqIn, BatchDataSeqIn, BatchSeqOut, BatchIntent, BatchDomain = vector2Tensor(BatchBERTSeqIn, BatchDataSeqIn,
                                                                                     BatchSeqOut, BatchIntent, BatchDomain)
            optimizer.zero_grad()

            outputs, slot_crf = model(BatchBERTSeqIn, BatchDataSeqIn, BatchSeqOut, lLensSeqin)
            outputDomain = outputs[0]
            outputIntent = outputs[1]
            outputSlots = outputs[2]

            BatchSeqOut = BatchSeqOut.view(BatchSeqOut.size(0) * BatchSeqOut.size(1))
            outputSlots = outputSlots.view(outputSlots.size(0) * outputSlots.size(1), SLOTSIZE)

            lossDomain   = criterionDomain(outputDomain, BatchDomain)
            lossIntent   = criterionIntent(outputIntent, BatchIntent)
            lossSlot     = criterionSlot(outputSlots, BatchSeqOut)

            epoch_lossDomain += lossDomain.item()
            epoch_lossIntent += lossIntent.item()
            epoch_lossSlot   += lossSlot.item()
            epoch_lossCRF    -= slot_crf.item()
    return (epoch_lossDomain / len(validIterator), epoch_lossIntent / len(validIterator), epoch_lossSlot / len(validIterator), epoch_lossCRF / len(validIterator))


if __name__ == '__main__':
    modelBest     = None
    model         = None
    optimizer     = None
    isTrainIntent = True
    isTrainSlot   = True
    lossMin       = 100
    ratioIntent   = 1
    ratioSlot     = 1

    for iter in range(TRAINITER):
        trainLoss, model, optimizer, dicts = train(iter, model=model, optimizer=optimizer, isTrainDomain=True, isTrainIntent=isTrainIntent, isTrainSlot=isTrainSlot, ratioIntent=ratioIntent, ratioSlot=ratioSlot)

        validLoss = evaluate(model, dicts)
        print("iter %d / %d: trainLoss = (domain=%f, intent=%f, slot=%f, crf=%f), validLoss = (domain=%f, intent=%f, slot=%f, crf=%f)" %
              (iter, TRAINITER, trainLoss[0], trainLoss[1], trainLoss[2], trainLoss[3], validLoss[0], validLoss[1], validLoss[2], validLoss[3]))

        if validLoss[2] < lossMin:
            lossMin = validLoss[2]
            modelBest = model
            save_model(modelBest, dicts, modelDir + "/complete", "complete.model", "complete.json")
            test()

