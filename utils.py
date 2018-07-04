# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import division
import pandas as pd
import tensorflow as tf
import sklearn.externals.joblib as joblib
import cPickle as pickle
import jieba
import random
import os
import codecs
import collections
import numpy as np
import re
import gensim
import jieba
from progressbar import ProgressBar
from sklearn.metrics import f1_score


PAD='<PAD>'
UNK='<UNK>'
TRAIN_DATA='ori_data/training.txt'
TEST_DATA='ori_data/validation.txt'

tag2id={'人类作者':0,'机器作者':1,'机器翻译':2,'自动摘要':3}
id2tag={0:'人类作者',1:'机器作者',2:'机器翻译',3:'自动摘要'}
tag2id_2={'人类作者':0,'自动摘要':1}
id2tag_2={0:'人类作者',1:'自动摘要'}

################################################################################
#                              Read data                                       #
################################################################################
def read_file(is_train,label_list=['人类作者','机器作者','机器翻译','自动摘要']):
    # 读取文件
    if is_train:
        df = pd.read_json(TRAIN_DATA, encoding='utf-8', lines=True)
        df = df.loc[df['标签'].isin(label_list)]
        x = df['内容'].values.tolist()
        y = df['标签'].values.tolist()
        return x,y
    else:
        df = pd.read_json(TEST_DATA, encoding='utf-8', lines=True)
        x = df['内容'].values.tolist()
        return x

def process(x,word_level=False):
    # 文本清洗,分词
    rNUM = '(-|\+)?\d+((\.|·)\d+)?%?'
    result=[]
    if word_level:
        def cut(x):
            return jieba.lcut(x,HMM=True)
    else:
        def cut(x):
            return [c for c in x]
    bar=ProgressBar(max_value=len(x))
    for sentence in bar(x):
        sentence=re.sub(r'\s','',sentence)
        sentence=re.sub(rNUM,'0',sentence)
        sentence=cut(sentence)
        result.append(sentence)
    return result

def load_data(is_train,word_level,label_list=['人类作者','机器作者','机器翻译','自动摘要']):
    # 加载预处理好的数据
    if is_train:
        if word_level:
            filename='data/train_w.pkl'
        else:
            filename='data/train_c.pkl'
        if os.path.exists(filename):
            print "Loading data from %s" % filename
            return pickle.load(open(filename, 'rb'))
        else:
            x,y=read_file(is_train=True,label_list=label_list)
            x=process(x,word_level)
            pickle.dump([x,y],open(filename,'wb'),pickle.HIGHEST_PROTOCOL)
            return x,y
    else:
        if word_level:
            filename='data/test_w.pkl'
        else:
            filename='data/test_c.pkl'
        if os.path.exists(filename):
            print "Loading data from %s" % filename
            return pickle.load(open(filename,'rb'))
        else:
            x=read_file(is_train=False)
            x=process(x,word_level)
            pickle.dump(x,open(filename,'wb'),pickle.HIGHEST_PROTOCOL)
            return x

def truncation(x):
    # 截断文本
    new_x = []
    for i in range(len(x)):
        new_x.append(x[i][:400])
    return new_x

################################################################################
#                              word2vec                                        #
################################################################################

def train_word2vec(sentences):
    """ 采用skip-gram 训练词向量 """
    model=gensim.models.Word2Vec(sentences=sentences,
                                 size=100,
                                 window=5,
                                 min_count=5,
                                 sg=1,
                                 workers=40,
                                 iter=5)
    model.wv.save_word2vec_format("data/word2vec_w.txt","data/vocab_w.txt",binary=True)

def load_word2vec(filename="data/word2vec_w.txt"):
    return gensim.models.KeyedVectors.load_word2vec_format(filename,binary=True)

def load_embeddings(word2id,filename="data/word2vec_w.txt"):
    print len(word2id)
    print max(word2id.values())
    pre_trained=gensim.models.KeyedVectors.load_word2vec_format(filename,binary=True)
    emb_dim=pre_trained.wv.syn0.shape[1]
    embeddings=np.zeros(shape=[len(word2id),emb_dim],dtype=np.float)
    for word,id in word2id.items():
        if word in pre_trained:
            embeddings[id] = pre_trained[word]
        else:
            embeddings[id] =np.random.uniform(-0.05,0.05,size=emb_dim)
    embeddings[word2id[UNK]] = np.random.uniform(-0.05, 0.05, size=emb_dim)
    embeddings[word2id[PAD]] = np.zeros(shape=emb_dim,dtype=np.float32)
    return embeddings


################################################################################
#                              Mapping                                         #
################################################################################

def build_vocab(x,y,min_df=5):
    # 创建词典
    words=[PAD,UNK]
    data=[]
    for sentence in x:
        for word in sentence:
            data.append(word)
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    for k,v in count_pairs:
        if v<min_df:
            break
        words.append(k)
    word2id ={word:id for id,word in enumerate(words)}
    id2word={id:word for id,word in enumerate(words)}
    y=list(set(y))
    tag2id = {tag:id for id,tag in enumerate(y)}
    id2tag= {id:tag for id,tag in enumerate(y)}
    return word2id,id2word,tag2id,id2tag

def build_x_ids(x,word2id):
    # mapping x to id
    ids=[[word2id.get(word,1) for word in sentence] for sentence in x]
    return ids

def build_y_ids(y,tag2id=tag2id):
    # mapping y to id
    ids=[tag2id[tag] for tag in y]
    return ids


################################################################################
#                              Batch Manager                                   #
################################################################################
def padding(sentences):
    maxlen=max([len(sentence) for sentence in sentences])
    pad_sentences=[]
    for sentence in sentences:
        pad=[0]*(maxlen-len(sentence))
        pad_sentences.append(sentence+pad)
    return pad_sentences


def minibatches(data, batch_size=2, is_train=True,shuffle=True):
    if is_train and shuffle:
        random.shuffle(data)
    num_batch=len(data)//batch_size
    for i in range(num_batch):
        batch_data=data[i*batch_size:(i+1)*batch_size]
        if is_train:
            x,y=zip(*batch_data)
            x=padding(x)
            yield x,y
        else:
            x=batch_data
            x=padding(x)
            yield x
    if is_train:
        batch_data=data[num_batch*batch_size:]
        x, y = zip(*batch_data)
        x = padding(x)
        yield x, y
    else:
        batch_data = data[num_batch * batch_size:]
        x = batch_data
        x = padding(x)
        yield x

################################################################################
#                              Evaluate                                        #
################################################################################
def score_all(y_true,y_pred,tag2id):
    # 计算全部类别的f1分数
    labels=[tag2id["人类作者"],tag2id['机器作者'],tag2id['机器翻译'],tag2id['自动摘要']]
    s1,s2,s3,s4=f1_score(y_true=y_true,y_pred=y_pred,labels=labels,average=None)
    print "人类作者:%f   机器作者:%f   机器翻译:%f   自动摘要:%f" % (s1,s2,s3,s4)
    f1=(s1+s2+s3+s4)/4
    return f1

def score2(y_true,y_pred,tag2id):
    # 只计算人类作者和自动摘要两个类别的f1分数
    labels = [tag2id["人类作者"],tag2id['自动摘要']]
    s1,s2= f1_score(y_true=y_true,y_pred=y_pred,labels=labels,average=None)
    print "人类作者:%f   自动摘要:%f" % (s1,s2)
    f1=(s1+s2)/2
    return f1

def score_eval_3(y_pred,dtrain):
    # xgboost eval函数，将人类作者和自动摘要合为一类
    y_true=dtrain.get_label()
    y_true=[x if x!=3 else 0 for x in y_true]
    y_pred=[x if x!=3 else 0 for x in y_pred]
    s1, s2, s3 = f1_score(y_true=y_true, y_pred=y_pred, labels=[0, 1, 2], average=None)
    print "人类作者/自动摘要:%f   机器作者:%f   机器翻译:%f  f1:%f" % (s1, s2, s3,(s1+s2+s3)/3)
    f1 = (s1 + s2 + s3) / 3
    return '1-f1_score', 1-f1

def score_eval_all(y_pred,dtrain):
    # xgboost eval函数
    y_true=dtrain.get_label()
    s1, s2, s3, s4 = f1_score(y_true=y_true, y_pred=y_pred, labels=[0, 1, 2, 3], average=None)
    print "人类作者:%f   机器作者:%f   机器翻译:%f   自动摘要:%f" % (s1, s2, s3, s4)
    f1 = (s1 + s2 + s3 + s4) / 4
    return '1-f1_score', 1-f1

################################################################################
#                              Results                                         #
################################################################################
def generate_result(y_pred,id2tag=id2tag,filename='result_nn.csv'):
    df = pd.read_json('ori_data/validation.txt', encoding='utf-8', lines=True)
    ids=df['id'].values.tolist()
    # 生成结果
    with codecs.open(filename,'w','utf-8') as f:
        for id,pred in zip(ids,y_pred):
            f.write(unicode(id)+','+id2tag[pred]+'\n')

def merge_result(filename1,filename2):
    # 合并二分类和四分类结果
    preds1=[]
    with codecs.open(filename1,'r','utf-8') as f1:
        for line in f1:
            pred=line.strip()[-4:]
            preds1.append(pred)
    preds2=[]
    with codecs.open(filename2,'r','utf-8') as f2:
        for line in f2:
            pred = line.strip()[-4:]
            preds2.append(pred)
    print len(preds1),len(preds2)
    assert len(preds1)==len(preds2)
    with codecs.open('final_result.csv','w','utf-8') as fout:
        for i in range(len(preds1)):
            if preds1[i]=='人类作者':
                fout.write(str(146421+i)+','+preds2[i]+'\n')
            else:
                fout.write(str(146421+i) + ',' + preds1[i] + '\n')

def error_print(y_pred,y_true,id2tag,x,id2word):
    # 错误分析
    lengths=[]
    error0=0
    error1=0
    with codecs.open('error_result.txt','w','utf-8') as f:
        for i in range(len(y_pred)):
            if y_pred[i]!=y_true[i]:
                if y_true[i]==0:
                    error0+=1
                else:
                    error1+=1
                lengths.append(len(x[i]))
                f.write(id2tag[y_pred[i]]+'\t')
                f.write(id2tag[y_true[i]]+'\t')
                f.write(''.join([id2word[c] for c in x[i]])+'\n')
    print lengths,id2tag[0],error0,id2tag[1],error1


if __name__ == '__main__':
    from multiprocessing import Process,Pool
    pool=Pool(processes=4)
    pool.apply_async(load_data,(True,True))
    pool.apply_async(load_data, (False, True))
    pool.apply_async(load_data, (True, False))
    pool.apply_async(load_data, (False, False))
    pool.close()
    pool.join()
    print "finished!"















