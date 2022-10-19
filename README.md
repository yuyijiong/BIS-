# BIS-labeling

BIS序列标注

2014_processed  是经过处理的训练集，已经转换为BIS标注
2014_processed_test  是经过处理的测试集
sgns.target.word-word.iter5 预训练词向量 语料来自百度百科（文件太大，不在此文件夹里）
预训练词向量 下载地址
https://pan.baidu.com/s/1Rn7LtTH0n7SHyHPfjRHbkg

task1.py 对应任务1，包含分词、关键词提取、word2vec找最相似词
pre.py 训练序列标注模型
use.py 测试序列标注模型
tools/ner_data_preprocess.py 预处理训练集的工具

model_label_sequence_CRF4355.h5 表示序列标注模型，用的CRF层，词汇表大小4355
word_index2729.pkl 表示词语到整数的映射字典，词汇表大小2729
label_index3.pkl 表示标签到整数的映射字典，标签表大小为3



