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
""" dataUtil.py """
COUNTWSIGN    =    3         # 词典特殊字符的数量
COUNTSSIGN    =    2         # 词槽特殊字符的数量        // SONLY_SIGN 不计入
COUNTISIGN    =    1         # 意图特殊字符的数量
WUNK_SIGN     =    0         # 未知词在词典的标号
WPAD_SIGN     =    1         # pad 在词典的编号
WEOS_SIGN     =    2         # 结束符标号
SUNK_SIGN     =    0         # 未知词槽标签编号
SPAD_SIGN     =    1         # pad 在词槽的编号
SONLY_SIGN    =    2         # 词槽‘O’对应的标签（也就是没有意义的词槽）
IUNK_SIGN     =    0         # 未知意图标签编号
SPAD          =  'O'         # 对于输入序列的<pad>，词槽对应的符号为O


""" train.py """
TRAINITER     =   20         # 迭代训练次数#
BATCHSIZE     =   16         # 切分出来的每个数据块的大小
WORDSIZE      =    0         # 词典大小
SLOTSIZE      =    0         # 词槽字典大小
INTENTSIZE    =    0         # 意图字典大小
EMBEDDSIZE    =  64         # 词向量大小     // 200
LSTMHIDSIZE   =  64         # LSTM隐含层大小 // 300
NLAYER        =    1         # LSTM的层数
DROPOUT       =  0.5         # dropout系数
MAXLEN        =   50         # 序列最大长度（训练时）
CLIP          =    1         # 最大梯度值
LEARNINGRATE  = 1e-2         # 学习速率
BIDIRECTIONAL = True         # 双向LSTM开关
MULTI_HIDDEN  =    2         # 双向LSTM对应隐藏层*2