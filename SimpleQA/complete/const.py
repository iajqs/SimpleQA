import sys

""" 路径 """
if sys.platform == "win32":
    trainDir = "C:/SimpleQA/data/atis/train"
    validDir = "C:/SimpleQA/data/atis/valid"
    testDir  = "C:/SimpleQA/data/atis/test"
    modelDir = "C:/SimpleQA/model"

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
COUNTDSIGN    =    1         # 领域特殊字符的数量
WPAD_SIGN     =    0         # pad 在词典的编号
WBUNK_SIGN    =  100         # 未知词在BERT词典的标号
WDUNK_SIGN    =    1         # 未知词在DATA词典的标号
WEOS_SIGN     =    2         # 结束符标号
SUNK_SIGN     =    0         # 未知词槽标签编号
SPAD_SIGN     =    1         # pad 在词槽的编号
IUNK_SIGN     =    0         # 未知意图标签编号
DUNK_SIGN     =    0         # 为止领域标签编号
MIN_WORD_COUNT=    2         # 词最小频率
WNUM          ="[NUM]"       # 所有数字转换对应字符串

""" train.py """
TRAINITER     =  100         # 迭代训练次数#
BATCHSIZE     =   64         # 切分出来的每个数据块的大小
WORDSIZE      =    0         # 词典大小
SLOTSIZE      =    0         # 词槽字典大小
INTENTSIZE    =    0         # 意图字典大小
BERTSIZE      =  768         # BERT编码层大小     // 768
ENCODERSIZE   =  768         # Encoder层输出大小  // 768
INTENTHIDSIZE =  768         # Intent层输入大小
NLAYER        =    1         # LSTM的层数
DROPOUT       =  0.2         # dropout系数
MAXLEN        =   21         # 序列最大长度（训练时）
CLIP          =  0.1         # 最大梯度值
LEARNINGRATE  = 1e-5         # 学习速率
ADDLENGTH     =    1         # 每个输入序列后追加的有效字符数量   // 一般追加结束符一个有效字符
BIDIRECTIONAL = True         # 双向LSTM开关
MULTI_HIDDEN  =    2         # 双向LSTM对应隐藏层*2