import keras
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras import utils
import os
import numpy as np
import keras.optimizers
import os
import tensorflow as tf
import pickle
from sklearn.metrics import accuracy_score
from keras_contrib.layers import CRF
from keras import metrics
from pre import save,load,load_word_label,load_text2tensor,decode_label


#使用已有的字典，把词映射为整数。字典需要和模型对应。
def wordlabel2tensor(sentences_list, label_list,word_index, label_index, onehot=1):
    # label 变成整数
    tokenizer2 = Tokenizer(num_words=None)
    tokenizer2.word_index=label_index  # 給每个词选中一个整数相对应
    label_sequences = tokenizer2.texts_to_sequences(label_list)  # 文本转换为整数序列，例如 我\的\狗 转换为 2\1\3

    # 生成语料词索引序列
    tokenizer = Tokenizer(num_words=None)  # 这个tokenizer对测试集也要用
    tokenizer.word_index=word_index  # 显示对应关系，即字典
    sequences = tokenizer.texts_to_sequences(sentences_list)  # 文本转换为整数序列，例如 我\的\狗 转换为 2\1\3

    # 全部pad到最长序列的长度
    MAX_SEQUENCE_LENGTH = len(max(sentences_list, key=len))
    data = pad_sequences(sequences, maxlen=100)
    label_sequences = pad_sequences(label_sequences, maxlen=100)

    # 标签格式转换

    if onehot==1:
        shape_ori=list(np.array(label_sequences).shape)
        shape_ori.append(-1)
        l_np=np.array(label_sequences)
        labels = utils.to_categorical(np.array(label_sequences).reshape(-1,1).squeeze(1))
        labels=labels.reshape(shape_ori)
    else:
        labels=np.array(label_sequences)

    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    x_train = data
    y_train = labels

    return x_train, y_train, labels.shape


if __name__ == '__main__':
    # 设置gpu内存自增长
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    #选择词汇表大小，不同的词汇表大小对应不同的模型，即不同的嵌入层大小
    vocabulary_size=4355
    #选择模型类型
    model_type=['dense','CRF']
    model_type=model_type[1]
    #加载对应词汇表的模型和字典
    #注意model和word—index必须编号相同才能一起用
    model=keras.models.load_model('model_label_sequence_'+model_type+str(vocabulary_size)+'.h5',compile = False,custom_objects={"CRF": CRF})
    word_index=load('word_index'+str(vocabulary_size)+'.pkl')
    label_index=load('label_index'+str(3)+'.pkl')

    # 处理数据，生成测试集
    word_all,label_all=load_word_label('2014_processed_test',100000)
    x_test,y_test,labels_shape =wordlabel2tensor(word_all,label_all,word_index,label_index)

    y_pred_d=model(x_test)

    test_result=np.array(metrics.categorical_accuracy(y_pred_d,y_test))
    acc_aver=test_result.sum()/test_result.size
    print("测试集平均准确率",acc_aver)

    #选择单个句子
    #想要选择第几句样本进行测试
    cho_num=1

    x_test1=x_test[cho_num,:]
    x_test1=np.expand_dims(x_test1,axis=0)
    #进行预测
    output=model(x_test1)

    #选出的句子
    cho_sent=word_all[cho_num]
    print("所选句子长度",len(cho_sent))

    #计算有零和没有零部分的准确率
    y_pred=np.argmax(output,axis=2)[0]
    y_true=np.argmax(y_test[cho_num],axis=1)

    acc1=accuracy_score(y_true,y_pred)
    acc2=accuracy_score(y_true[-len(cho_sent):],y_pred[-len(cho_sent):])
    print("包含0时准确率",acc1)
    print("不包含0时准确率",acc2)

    #把输出的向量转化为 bis标签，l固定100长度
    l=decode_label(output,label_index)

    #真实的标签
    cho_label = label_all[cho_num]
    cho_label=list(map(lambda x:x.lower(),cho_label)) #转换为小写

    #打印结果
    if len(cho_sent)>100:
        #所选句子大于100，则只取后面100字
        cho_sent=cho_sent[-100:]
        ll=cho_label[-100:]
        print("原始样本", list(zip(cho_sent, ll)))
        print("模型预测", list(zip(cho_sent, l)))
    else:
        # 所选句子小于100，则l只取后面不为零的部分
        print("原始样本",list(zip(cho_sent,cho_label)))
        print("模型预测",list(zip(cho_sent,l[-len(cho_sent):])))