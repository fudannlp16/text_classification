# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import division
import numpy as np
import lightgbm as lgb
import utils
import os
import tensorflow as tf
import sklearn.externals.joblib as joblib
import cPickle as pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

tf.flags.DEFINE_boolean("train",True,"is_train")
tf.flags.DEFINE_boolean("word_level",True,"word_level")
FLAGS=tf.flags.FLAGS

def train():
    if FLAGS.word_level:
        vec_file='temp/ml/vec_w_tfidf.pkl'
        model_file='temp/ml/xgb_model_w.bin'
    else:
        vec_file = 'temp/ml/vec_c_tfidf.pkl'
        model_file = 'temp/ml/xgb_model_c.bin'

    x, y = utils.load_data(True, FLAGS.word_level)
    x, y = x[:20000], y[:20000]
    x = [' '.join(s) for s in x]
    y = utils.build_y_ids(y)

    # 划分验证集
    x_train, x_dev, y_train, y_dev=train_test_split(x,y,test_size=10000,random_state=24)

    # 特征抽取
    vectorizer = TfidfVectorizer(lowercase=False,analyzer='word',min_df=100,max_df=0.9,
                                 ngram_range=(1,2),tokenizer=lambda x:x.split())
    vectorizer.fit(x_train)
    # joblib.dump(vectorizer,vec_file)

    # 句子表示
    x_train=vectorizer.transform(x_train)
    x_dev=vectorizer.transform(x_dev)

    print x_train.shape,x_dev.shape

    lgb_train=lgb.Dataset(x_train,y_train)
    lgb_dev=lgb.Dataset(x_dev,y_dev,reference=lgb_train)


    param = {'max_depth': 6, 'application':'multiclass',
             'num_class':4,'num_threads':40}

    print 'Train started!'
    bst=lgb.train(params=param,
                  train_set=lgb_train,
                  num_boost_round=500,
                  valid_sets=lgb_dev,
                  early_stopping_rounds=10,
                  verbose_eval=True)
    bst.save_model(model_file,num_iteration=bst.best_iteration) # 保存模型

    print 1-bst.best_score, bst.best_iteration

if __name__ == '__main__':
    if FLAGS.train==1:
        train()
    else:
        test()








