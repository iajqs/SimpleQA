"""
通过上面2个实例，我们发现编写pytest测试样例非常简单，只需要按照下面的规则：

测试文件以test_开头（以_test结尾也可以）
测试类以Test开头，并且不能带有 init 方法
测试函数以test_开头
断言使用基本的assert即可

作者：呆呆冬
链接：https://www.jianshu.com/p/932a4d9f78f8
来源：简书
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

"""
import sys
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
import pytest

""" dataUtil.py """
class TestDataUtil:
    def test_getData(self):
        dataSeqIn, dataSeqOut, dataIntent = getData(trainDir)
        print(len(dataSeqIn))
        print(dataSeqIn[0:2])
        print(dataSeqOut[0:2])
        assert len(dataSeqIn) == len(dataSeqOut) == len(dataIntent)

    def test_getWordDictionary(self):
        dataSeqIn = [['i', 'want', 'to', 'fly', 'from', 'baltimore', 'to', 'dallas', 'round', 'trip']]
        dictWord = getWordDictionary(dataSeqIn)
        print(dictWord[0])
        print(dictWord[1])

    def test_getSlotDictionary(self):
        dataSeqIn, dataSeqOut, dataIntent = getData(trainDir)

        dictSlot = getSlotDictionary(dataSeqOut)
        for item in dictSlot[0]:
            idx = dictSlot[0][item]
            print(item, ":", idx, ';\t', dictSlot[1][idx], ":", idx)
            assert item == dictSlot[1][idx]

    def test_getIntentDictionary(self):
        dataSeqIn, dataSeqOut, dataIntent = getData(trainDir)
        dictIntent = getIntentDictionary(dataIntent)
        for item in dictIntent[0]:
            idx = dictIntent[0][item]
            print(item, ":", idx, ';\t', dictIntent[1][idx], ":", idx)
            assert item == dictIntent[1][idx]

    def test_normalizeString(self):
        s1 = "flights! from cincinnati to o'hare departing after 718 am american?"
        s2 = "O O O O O B-fromloc.city_name O B-depart_time.time I-depart_time.time O O O B-toloc.city_name O B-arrive_time.time O O B-arrive_time.period_of_day"
        print()
        print(s1)
        print(normalizeString(s1))
        print(normalizeString(s1.strip()).split(' '))
        print(s2)
        print(normalizeString(s2))
        print(normalizeString(s2.strip()).split(' '))

    def test_makePairs(self):
        print()
        dataSeqIn, dataSeqOut, dataIntent = getData(trainDir)
        pairs = makePairs(dataSeqIn, dataSeqOut, dataIntent)  # 根据原数据生成样例对    zip(dataSeqIn, dataSeqOut, dataIntent)

        items = pairs[0]
        for index, word in enumerate(items[0]):
            assert word == dataSeqIn[0][index]
        for index, slot in enumerate(items[1]):
            assert slot == dataSeqOut[0][index]
        assert items[2] == dataIntent[0]


        for item in items:
            print(item)
        print(dataSeqIn[0], "\n", dataSeqOut[0], "\n", dataIntent[0])

    def test_transIds(self):
        ''' 读取数据 '''
        dataSeqIn, dataSeqOut, dataIntent = getData(trainDir)   # 获取原数据
        dictWord   = getWordDictionary(dataSeqIn)               # 获取词典  (word2index, index2word)
        dictSlot   = getSlotDictionary(dataSeqOut)              # 获取词槽标签字典  (slot2index, index2slot)
        dictIntent = getIntentDictionary(dataIntent)            # 获取意图标签字典  (intent2index, index2intent)
        pairs      = makePairs(dataSeqIn, dataSeqOut, dataIntent)                   # 根据原数据生成样例对    zip(dataSeqIn, dataSeqOut, dataIntent)
        pairsIded  = transIds(pairs, dictWord[0], dictSlot[0], dictIntent[0])       # 将字词都转换为数字id

        items = pairsIded[0]
        for index, wordID in enumerate(items[0]):
            assert dictWord[1][wordID] == dataSeqIn[0][index]
            assert wordID == dictWord[0][dataSeqIn[0][index]]
        for index, slotID in enumerate(items[1]):
            assert dictSlot[1][slotID] == dataSeqOut[0][index]
            assert slotID == dictSlot[0][dataSeqOut[0][index]]
        assert dictIntent[1][items[2]] == dataIntent[0]
        assert items[2] == dictIntent[0][dataIntent[0]]
        for item in items:
            print(item)
        for item in dataSeqIn[0]:
            print(dictWord[0][item], end=" ")
        print()
        for item in dataSeqOut[0]:
            print(dictSlot[0][item], end=" ")
        print()
        print(dataSeqIn[0], "\n", dataSeqOut[0], "\n", dataIntent[0])

    def test_splitData(self):
        print()
        dataSeqIn, dataSeqOut, dataIntent = getData(trainDir)  # 获取原数据
        dictWord    = getWordDictionary(dataSeqIn)  # 获取词典  (word2index, index2word)
        dictSlot    = getSlotDictionary(dataSeqOut)  # 获取词槽标签字典  (slot2index, index2slot)
        dictIntent  = getIntentDictionary(dataIntent)  # 获取意图标签字典  (intent2index, index2intent)
        pairs       = makePairs(dataSeqIn, dataSeqOut, dataIntent)  # 根据原数据生成样例对    zip(dataSeqIn, dataSeqOut, dataIntent)
        pairsIded   = transIds(pairs, dictWord[0], dictSlot[0], dictIntent[0])  # 将字词都转换为数字id
        trainIterator = splitData(pairsIded)  # 讲样例集按BATCHSIZE大小切分成多个块

        for item in trainIterator:
            print(len(item[0]))

    def test_getMaxLengthFromBatch(self):
        print()
        batch = [[[1, 2, 3, 4, 5, 6, 7], [1] * 60, [1, 2, 3], [44, 2, 4, 5]], [], []]
        assert getMaxLengthFromBatch(batch, 2) == 62

    def test_getSeqInLengthsFrombatch(self):
        print()
        batch = [[[1, 2, 3, 4, 5, 6, 7], [1] * 60, [1, 2, 3], [44, 2, 4, 5]], [], []]
        targets = [8, 61, 4, 5]
        lengths = getSeqInLengthsFromBatch(batch, 1)
        for index, length in enumerate(lengths):
            assert length == targets[index]

    def test_padBatch(self):
        print()
        batch = [[[1, 2, 3, 4, 5, 6, 7], [1] * 2, [1, 2, 3], [44, 2, 4, 5]], [[4, 2,12,44, 2],[],[],[]], [1, 2, 3]]
        MAXLEN = getMaxLengthFromBatch(batch, ADDLENGTH)
        batch = padBatch(batch, MAXLEN_TEMP=MAXLEN)

        for item in batch[0]:
            print(item)
        for item in batch[1]:
            print(item)
        # print(batch)

    def test_vector2Tensor(self):
        print()
        batch = [[[1, 2, 3, 4, 5, 6, 7], [1] * 2, [1, 2, 3], [44, 2, 4, 5]], [[4, 2,12,44, 2],[],[],[]], [1, 2, 3]]
        MAXLEN = getMaxLengthFromBatch(batch, ADDLENGTH)
        batch = padBatch(batch, MAXLEN_TEMP=MAXLEN)
        BatchSeqIn  = batch[0]          # 文本序列
        BatchSeqOut = batch[1]          # 词槽标签序列
        BatchIntent = batch[2]          # 意图标签
        BatchSeqIn, BatchSeqOut, BatchIntent = vector2Tensor(BatchSeqIn, BatchSeqOut, BatchIntent)
        print(BatchSeqIn)
        print(BatchSeqOut)
        print(BatchIntent)


import torch

class TestModel:
    def test_EncoderRNN(self):
        WORDSIZE = 20

        inputs = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=torch.long, device="cpu")
        encoder = EncoderRNN(input_size=WORDSIZE, emb_size=EMBEDDSIZE, pading_idx=WPAD_SIGN, hidden_size=LSTMHIDSIZE,
                             n_layers=NLAYER, dropout=DROPOUT, bidirectional=BIDIRECTIONAL)

        result = encoder(inputs)
        print(result[0])
