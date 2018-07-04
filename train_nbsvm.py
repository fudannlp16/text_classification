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
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.linear_model import LogisticRegression
from scipy import sparse
from sklearn.model_selection import train_test_split


tf.flags.DEFINE_boolean("train",True,"is_train")
tf.flags.DEFINE_boolean("word_level",True,"word_level")
FLAGS=tf.flags.FLAGS

class NbSvmClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, dual=False, n_jobs=1):
        self.C = C
        self.dual = dual
        self.n_jobs = n_jobs

    def predict(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict(x.multiply(self._r))

    def predict_proba(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict_proba(x.multiply(self._r))

    def fit(self, x, y):
        # Check that X and y have correct shape
        x, y = check_X_y(x, y, accept_sparse=True)

        def pr(x, y_i, y):
            p = x[y==y_i].sum(0)
            return (p+1) / ((y==y_i).sum()+1)

        self._r = sparse.csr_matrix(np.log(pr(x,1,y) / pr(x,0,y)))
        x_nb = x.multiply(self._r)
        self._clf = LogisticRegression(C=self.C, dual=self.dual, n_jobs=self.n_jobs).fit(x_nb, y)
        return self

def train():
    if FLAGS.word_level:
        vec_file='temp/ml/vec_w_tfidf.pkl'
        model_file='temp/ml/nbsvm_model_w.bin'
    else:
        vec_file = 'temp/ml/vec_c_tfidf.pkl'
        model_file = 'temp/ml/nbsvm_model_c.bin'

    x, y = utils.load_data(True, FLAGS.word_level)
    x = [' '.join(s) for s in x]
    y = utils.build_y_ids(y)

    x_test = utils.load_data(False, FLAGS.word_level)
    x_test = [' '.join(s) for s in x_test]

    # 划分验证集
    x_train, x_dev, y_train, y_dev=train_test_split(x,y,test_size=10000,random_state=24)

    # 特征抽取
    vectorizer = TfidfVectorizer(lowercase=False,analyzer='word',min_df=100,max_df=0.9,
                                 ngram_range=(1,2),tokenizer=lambda x:x.split())
    vectorizer.fit(x_train)

    # 句子表示
    x_train=vectorizer.transform(x_train)
    x_dev=vectorizer.transform(x_dev)
    x_test=vectorizer.transform(x_test)

    joblib.dump(x_test,vec_file)

    print x_train.shape, x_dev.shape

    model = NbSvmClassifier(C=4, dual=True, n_jobs=-1)
    model.fit(x_train,y_train)
    joblib.dump(model,model_file)

    preds=model.predict(x_dev)
    f1=utils.score_all(y_dev,preds,utils.tag2id)
    print f1

def test():
    if FLAGS.word_level:
        vec_file='temp/ml/vec_w_tfidf.pkl'
        model_file='temp/ml/nbsvm_model_w.bin'
    else:
        vec_file = 'temp/ml/vec_c_tfidf.pkl'
        model_file = 'temp/ml/nbsvm_model_c.bin'

    x=joblib.load(vec_file)
    model=joblib.load(model_file)
    predicts = model.predict(x)
    utils.generate_result(y_pred=predicts, filename='result_xgb.csv')



if __name__ == '__main__':
    if FLAGS.train==1:
        train()
    else:
        test()
