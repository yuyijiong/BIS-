import jieba
import jieba.analyse
import gensim
import jieba.posseg

#1
#读取文件
f = open('wiki.txt', mode='r',encoding='utf-8')
txt = f.read(100000)
#先按空格分句
txt_sen=txt.split(' ')
#每句分词
result_split=[]
for sent in txt_sen:
    #t=jieba.posseg.lcut(sent)#分词+词性标注
    t = jieba.lcut(sent,cut_all=False,HMM=True)
    result_split.append(t)

print('分词结果',result_split)

#2
#提取关键词
a1=jieba.analyse.extract_tags(txt,topK=20, withWeight=False,allowPOS=(['nz', 'vn', 'n','nr']))
a2=jieba.analyse.textrank(txt,topK=20, withWeight=False,allowPOS=(['nz', 'vn', 'n','nr']))
print('关键词提取结果 extract_tags  ',a1)
print('关键词提取结果 textrank  ',a2)


#3
#构建语料库
sen=gensim.models.word2vec.Text8Corpus('wiki.txt')#可以直接从txt文件构建语料库

#训练模型，把上一题分词结果的列表作为语料
model=gensim.models.word2vec.Word2Vec(result_split, vector_size=10,window=10,min_count=1,sg=0) #sg=0为cbow

#找最大相似词
y1 = model.wv.similar_by_word(u"数学家", topn=10)
y2 = model.wv.similar_by_word(u"历史学家", topn=10)

print('最大相似词',y1)

