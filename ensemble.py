# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import division
import numpy as np
import tensorflow as tf
import utils
import nn_models as models
import os
from progressbar import ProgressBar

class Graph(object):
    """ Create model graph """
    def __init__(self, model_name, model_file, vocab_size, emb_dim, num_classes):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            self.model = getattr(models, model_name)(vocab_size, emb_dim, num_classes)
            saver = tf.train.Saver(tf.global_variables())
            model_file = os.path.join('temp', 'nn', model_file)
            saver.restore(self.sess, model_file)
    def run(self, data):
        predicts = []
        bar = ProgressBar(max_value=len(data) // 1024 + 1)
        for batch_data in bar(utils.minibatches(data, 1024, False)):
            predict = self.model.eval_step(self.sess, batch_data)
            predicts.extend(predict)
        print 'The model is finished!'
        return predicts

def ensemble(predicts):
    # ensemble
    predicts = [np.array(predict) for predict in predicts]
    predicts = np.stack(predicts,axis=2)
    predicts = np.average(predicts, axis=-1)
    predicts = np.argmax(predicts, axis=-1)
    results = predicts.tolist()
    return results

def ensemble2(predicts):
    predicts=[np.argmax(np.array(predict),axis=1) for predict in predicts]
    predicts = np.stack(predicts,axis=1)
    predicts = np.average(predicts,axis=-1)
    results = [1 if predict>0.5 else 0 for predict in predicts]
    return results

def predict():
    ################################################################################
    #                              NN model                                        #
    ################################################################################
    x, y = utils.read_file(is_train=True,label_list=['人类作者','自动摘要','机器作者','机器翻译'])
    x = utils.process(x)
    x = utils.truncation(x)
    word2id,id2word,tag2id,id2tag=utils.build_vocab(x,y,min_df=10)

    test_x=utils.read_file(is_train=False)
    test_x = utils.process(test_x)
    test_x = utils.truncation(test_x)
    test_x = utils.build_x_ids(test_x,word2id)


    vocab_size=len(word2id)
    emb_dim=100
    num_classes=len(tag2id)

    print "测试集数据大小:%d" % (len(test_x))
    print "vocab_size:%d num_classes:%d" % (vocab_size,num_classes)

    results=[]
    g1 = Graph('TextCNN', 'HierarchyCNN',vocab_size,emb_dim,num_classes)
    results.append(g1.run(test_x))

    ################################################################################
    #                              Other model                                     #
    ################################################################################



    ################################################################################
    #                              Ensemble                                       #
    ################################################################################
    final_result=ensemble(results)
    utils.generate_result(final_result,id2tag,'result_nn.csv')

if __name__ == '__main__':
    predict()
