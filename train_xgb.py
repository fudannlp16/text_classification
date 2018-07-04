# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import division
import numpy as np
import xgboost as xgb
import utils
import os
import tensorflow as tf
import sklearn.externals.joblib as joblib
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

    x = [' '.join(s) for s in x]
    y = utils.build_y_ids(y)

    x_test = utils.load_data(False, FLAGS.word_level)
    x_test = [' '.join(s) for s in x_test]

    # 划分验证集
    x_train, x_dev, y_train, y_dev=train_test_split(x,y,test_size=10000,random_state=24)

    # 特征抽取
    vectorizer = TfidfVectorizer(lowercase=False,analyzer='word',min_df=100,max_df=1.0,
                                 ngram_range=(1,2),tokenizer=lambda x:x.split())
    vectorizer.fit(x_train)

    # 句子表示
    x_train=vectorizer.transform(x_train)
    x_dev=vectorizer.transform(x_dev)
    x_test=vectorizer.transform(x_test)

    joblib.dump(x_test,vec_file)

    print x_train.shape,x_dev.shape

    dtrain=xgb.DMatrix(data=x_train,label=y_train)
    ddev=xgb.DMatrix(data=x_dev,label=y_dev)

    param = {'max_depth': 6, 'objective':'multi:softmax',
             'num_class':4,'nthread':40,'silent':1}
    evallist = [(dtrain, 'train'),(ddev, 'eval')]

    print 'Train started!'
    bst=xgb.train(params=param,
                  dtrain=dtrain,
                  num_boost_round=500,
                  evals=evallist,
                  early_stopping_rounds=20,
                  verbose_eval=True,
                  feval=utils.score_eval_all)
    bst.save_model(model_file) # 保存模型

    print 1-bst.best_score, bst.best_iteration


def test():
    if FLAGS.word_level:
        vec_file='temp/ml/vec_w_tfidf.pkl'
        model_file='temp/ml/xgb_model_w.bin'
    else:
        vec_file = 'temp/ml/vec_c_tfidf.pkl'
        model_file = 'temp/ml/xgb_model_c.bin'

    x=joblib.load(vec_file)

    print x.shape
    dtest=xgb.DMatrix(data=x)

    bst_new = xgb.Booster({'nthread': 40})  # init model
    bst_new.load_model(model_file)  # load data

    print 'Predict started!'
    predicts=bst_new.predict(dtest,ntree_limit=176)
    utils.generate_result(y_pred=predicts,filename='result_xgb.csv')

if __name__ == '__main__':
    if FLAGS.train==1:
        train()
    else:
        test()








