# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import division
from sklearn.model_selection import train_test_split
import tensorflow as tf
import utils
import codecs
import random
import fastText
import os
import utils

tf.flags.DEFINE_boolean("train",True,"is_train")
tf.flags.DEFINE_boolean("word_level",True,"word_level")
FLAGS=tf.flags.FLAGS


def generate_file():
    #生成fasttext 训练集,验证集
    x,y=utils.load_data(True,True)
    x_train, x_dev, y_train, y_dev = train_test_split(x, y, test_size=10000, random_state=24)
    with codecs.open('data/train_w.fast','w','utf-8') as f:
        for i in range(len(x_train)):
            f.write('__label__')
            f.write(y_train[i]+' ')
            f.write(' '.join(x_train[i])+'\n')
    with codecs.open('data/dev_w.fast','w','utf-8') as f:
        for i in range(len(x_dev)):
            f.write('__label__')
            f.write(y_dev[i]+' ')
            f.write(' '.join(x_dev[i])+'\n')

    x,y=utils.load_data(True,False)
    x_train, x_dev, y_train, y_dev = train_test_split(x, y, test_size=10000, random_state=24)
    with codecs.open('data/train_c.fast','w','utf-8') as f:
        for i in range(len(x_train)):
            f.write('__label__')
            f.write(y_train[i]+' ')
            f.write(' '.join(x_train[i])+'\n')
    with codecs.open('data/dev_c.fast','w','utf-8') as f:
        for i in range(len(x_dev)):
            f.write('__label__')
            f.write(y_dev[i]+' ')
            f.write(' '.join(x_dev[i])+'\n')

def train():
    if FLAGS.word_level:
        train_file='data/train_w.fast'
        dev_file = 'data/dev_w.fast'
        model_file='temp/ml/fast_model_w.bin'
    else:
        train_file='data/train_c.fast'
        dev_file = 'data/dev_c.fast'
        model_file='temp/ml/fast_model_c.bin'
    model=fastText.train_supervised(
        input=train_file,
        dim=100,
        epoch=15,
        thread=40,
        minCount=10,
        loss='softmax',
        wordNgrams=2
    )
    model.save_model(model_file)

    vocab_size,p,r=model.test(train_file)
    f1=(p*r*2)*1.0/(p+r)
    print 'Train:vocab_size:%d  p:%.5f   r:%.5f   f1:%.5f' % (vocab_size,p,r,f1)

    vocab_size,p,r=model.test(dev_file)
    f1=(p*r*2)*1.0/(p+r)
    print 'Dev:vocab_size:%d  p:%.5f   r:%.5f   f1:%.5f' % (vocab_size,p,r,f1)


def test():
    if FLAGS.word_level:
        x=utils.load_data(False,True)
        model_file = 'temp/ml/fast_model_w.bin'
    else:
        x=utils.load_data(False,False)
        model_file = 'temp/ml/fast_model_c.bin'
    x=[' '.join(sentence) for sentence in x]
    model = fastText.load_model(model_file)
    preds=[]
    for sentence in x:
        labels,prob=model.predict(sentence,k=1)
        preds.append(utils.tag2id[labels[0][-4:]])
    utils.generate_result(y_pred=preds,filename='result_fast.csv')


if __name__ == '__main__':
    # generate_file()
    if FLAGS.train==1:
        train()
    else:
        test()


