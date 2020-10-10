import sys

""" 路径 """
if sys.platform == "win32":
    trainDir = "D:/SimpleQA/data/atis/train"
    validDir = "D:/SimpleQA/data/atis/valid"
    testDir  = "D:/SimpleQA/data/atis/test"
    modelDir = "D:/SimpleQA/model"
else:
    trainDir = "/home/cks/program/SimpleQA/data/atis/train"
    validDir = "/home/cks/program/SimpleQA/data/atis/valid"
    testDir  = "/home/cks/program/SimpleQA/data/atis/test"
    modelDir = "/home/cks/program/SimpleQA/model"

""" 设定模型超参数 """
TRAINITER     =   20         # 迭代训练次数#
BATCHSIZE     =  128         # 切分出来的每个数据块的大小
PAD_IDX       =    0         # pad 在词典的下标
EOS_IDX       =    1         # 结束符下标
WORDSIZE      =    0         # 词典大小
SLOTSIZE      =    0         # 词槽字典大小
INTENTSIZE    =    0         # 意图字典大小
EMBEDDSIZE    =  128         # 词向量大小
LSTMHIDSIZE   =  128         # LSTM隐含层大小
NLAYER        =    1         # LSTM的层数
DROPOUT       =  0.1         # dropout系数
MAXLEN        =   20         # 序列最大长度（训练时）
CLIP          =    1         # 最大梯度值
LEARNINGRATE  = 1e-3         # 学习速率
BIDIRECTIONAL = True         # 双向LSTM开关
MULTI_HIDDEN  = 2            # 双向LSTM对应隐藏层*2