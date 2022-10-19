import keras
import keras_metrics
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras import utils
import os
import numpy as np
import keras.optimizers
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras_contrib.layers import CRF
from keras import losses
from keras.callbacks import EarlyStopping
from tensorflow.python.client import device_lib
import os
import tensorflow as tf
import pickle
from keras import metrics



def save(v,filename):
  f=open(filename,'wb')
  pickle.dump(v,f)
  f.close()
  return filename

def load(filename):
  f=open(filename,'rb')
  r=pickle.load(f)
  f.close()
  return r

#从预处理过得文件中，提取出 句子 和 BIS标签
def load_word_label(dir,leng=-1):
    f1 = open(dir, mode='r', encoding='utf-8')

    d=f1.read(leng)

    d2=d.split('\n')

    sentence_list=[]
    label_list=[]
    for d3 in d2:
        if len(d3)<300:
            continue
        try:words,labels=d3.split('\t')
        except:continue
        #提取单个字的列表，和BIS列表

        word=words.split(' ')
        labels_sp=labels.split(' ')
        while '' in labels_sp:
            labels_sp.remove('')

        labels_BIS=list(map(lambda x:x[0],labels_sp))
        #数目相等才写入
        if len(word)==len(labels_BIS):
            sentence_list.append(word)
            label_list.append(labels_BIS)
        else:continue

    print("最短句子的长度",len(min(sentence_list,key=len)))

    print("样本总数",len(sentence_list))
    return sentence_list,label_list

#把（word列表，标签列表）转化为 （整数序列，整数序列）
def load_text2tensor(sentences_list,label_list):
    #label 变成整数
    tokenizer2 = Tokenizer(num_words=None)
    tokenizer2.fit_on_texts(label_list) #給每个词选中一个整数相对应
    label_sequences = tokenizer2.texts_to_sequences(label_list)#标签转换为整数序列，例如 b\i\s 转换为 2\1\3

    label_index = tokenizer2.word_index   #显示对应关系，即字典。使用测试集时，也要用这个字典把词转换为整数。
    '''
    label_index={}
    label_index['b']=0
    label_index['i']=1
    label_index['s']=2
    '''

    # 生成语料词索引序列
    tokenizer = Tokenizer(num_words=None) #这个tokenizer对测试集也要用
    tokenizer.fit_on_texts(sentences_list) #給每个词选中一个整数相对应
    sequences = tokenizer.texts_to_sequences(sentences_list)#文本转换为整数序列，例如 我\的\狗 转换为 2\1\3

    word_index = tokenizer.word_index   #显示对应关系，即字典
    print('词汇表大小' , len(word_index))

    #全部pad到最长序列的长度(在前面加0)
    MAX_SEQUENCE_LENGTH=len(max(sentences_list, key=len))
    MAX_SEQUENCE_LENGTH=100
    data =pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    label_sequences =pad_sequences(label_sequences, maxlen=MAX_SEQUENCE_LENGTH)

    # 标签格式转换
    onehot=1
    if onehot==1:
        shape_ori=list(np.array(label_sequences).shape)
        shape_ori.append(-1)
        labels = utils.to_categorical(np.array(label_sequences).reshape(-1,1).squeeze(1))
        labels=labels.reshape(shape_ori)
    else:
        labels=np.array(label_sequences)

    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    # 切分成测试集和验证集
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    nb_validation_samples = int(0.1 * data.shape[0])

    x_train = data[:-nb_validation_samples]
    y_train = labels[:-nb_validation_samples]
    x_val = data[-nb_validation_samples:]
    y_val = labels[-nb_validation_samples:]

    return x_train,y_train,x_val,y_val,labels.shape,word_index,label_index

#构建嵌入层
def cons_embed(word_index,MAX_SEQUENCE_LENGTH):
    # word_index  输入语料的每个词对应的整数索引
    # 使用已有词向量
    embeddings_index = {}  # 这个字典记录（word，词向量）,把预训练文件转化为字典
    pretrained_wv_dir = "sgns.target.word-word.iter5"
    f = open(pretrained_wv_dir, mode='r', encoding='utf-8')
    for line in f:
        values = line.split()

        # 因为每行的第一个元素是词，后面的才是词向量，因此将values[0]与values[1:]分开存放
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')

        embeddings_index[word] = coefs  # 建立字典

    f.close()

    # embedding_matrix的维度 =（语料单词的数量，预训练词向量的维度），行数取决于词汇表大小
    EMBEDDING_DIM=300
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))

    #逐行把矩阵里填上词向量，wordindex从是1开始，所以矩阵第0行全为0，用于pad
    for word, i in word_index.items():
        # 根据语料生成的word_index来从embedding_index一条条获取单词对应的向量
        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:
            # words found in embedding index will be pretrained vectors.
            embedding_matrix[i] = embedding_vector   # i+1 是为了处理OOV，使得预测时未见过的词的位置为0。当然如果不使用这种OOV的处理方式的话，这里的embedding_matrix[i+1]应该变成embedding_matrix[i]，下同理。
        else:
            # words not found in embedding index will be random vectors with certain mean&std.如果单词未能在预训练词表中找到，可以自己生成一串向量。
            embedding_matrix[i] = np.random.normal(0.053, 0.3146, (1, 300))[0] # 0.053, 0.3146 根据统计，可以改变数值

    from keras.layers import Embedding
    print("嵌入矩阵大小",embedding_matrix.shape)
    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)

    return embedding_layer

#把整数解码为标签bis
def decode_label(model_output,label_index):

    label=np.argmax(model_output, axis=2)
    label=label[0].tolist()
    #my_dic = {k: v for v, k in label_index.items()}
    label_list=list(label_index.keys())
    for i in range(len(label)):
        label[i]=label_list[label[i]-1]

    return label

#构建模型
def buildmodel():
    #读取或构建嵌入层
    if os.path.exists('embed_layer'+str(len(word_index))+'.pkl'):
        print('读取嵌入层')
        embed_layer = load('embed_layer'+str(len(word_index))+'.pkl')
    else:
        print('建立嵌入层')
        embed_layer = cons_embed(word_index, MAX_SEQUENCE_LENGTH)
        save(embed_layer, 'embed_layer'+str(len(word_index))+'.pkl')

    model = Sequential()
    model.add(embed_layer)
    model.add(Bidirectional(LSTM(100, return_sequences=True)))
    model.add(Bidirectional(LSTM(150, return_sequences=True)))
    #model.add(CRF(4, learn_mode='marginal', sparse_target=True))
    model.add(Dense(4,activation="softmax"))
    model.summary()
    opt = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=[metrics.categorical_accuracy])
    return model

if __name__ == '__main__':
    # 设置gpu内存自增长
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 数字为gpu编号

    #处理数据，生成训练集
    word_all,label_all=load_word_label('2014_processed',10000000)
    x_train,y_train,x_val,y_val,labels_shape,word_index,label_index =load_text2tensor(word_all,label_all)

    #保存字典，测试时要用到
    save(word_index,'word_index'+str(len(word_index))+'.pkl')
    save(label_index,'label_index'+str(len(label_index))+'.pkl')
    print(label_index)

    #计算样本总数，最大序列长度，标签类型的个数（几种词性）
    sample_num=labels_shape[0]
    MAX_SEQUENCE_LENGTH=labels_shape[1]
    label_type_num=labels_shape[2]

    #建立模型并训练
    #选择模型类型
    model_type=['dense','CRF']
    model_type=model_type[0]

    #读取或构建嵌入层
    if os.path.exists('model_label_sequence_'+model_type+str(len(word_index))+'.h5'):
        print('读取模型')
        model = keras.models.load_model('model_label_sequence_' + model_type + str(len(word_index)) + '.h5',
                                        custom_objects={"CRF": CRF})
    else:
        print('建立模型')
        model = buildmodel()

    model.fit(x_train, y_train, batch_size=100, epochs=3)
    model.save('model_label_sequence_'+model_type+str(len(word_index))+'.h5')




