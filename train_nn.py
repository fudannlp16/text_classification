# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import division
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import utils
import nn_models as models
from progressbar import ProgressBar
from sklearn.model_selection import train_test_split

tf.flags.DEFINE_integer('batch_size',512,'batch_size')
tf.flags.DEFINE_integer("max_epoch",100,'max_epoch')
tf.flags.DEFINE_string("model_name","TextCNN","model_name")
tf.flags.DEFINE_string("model_file",'TextCNN',"model_file")
tf.flags.DEFINE_string("output_file",'result_nn.csv','output_file')
tf.flags.DEFINE_boolean("train",True,"is_train")

FLAGS=tf.flags.FLAGS

def train():
    x,y=utils.load_data(True,True)

    word2id,id2word,tag2id,id2tag=utils.build_vocab(x,y,min_df=20)

    x = utils.build_x_ids(x,word2id)
    y = utils.build_y_ids(y,tag2id)
    data=zip(x,y)

    train_data,dev_data =train_test_split(data,test_size=10000,random_state=24)

    #pre_embeddings=utils.load_embeddings(word2id)

    vocab_size=len(word2id)
    emb_dim=100
    num_classes=len(tag2id)

    print "训练集数据大小:%d 验证集数据大小:%d" % (len(train_data),len(dev_data))
    print "vocab_size:%d num_classes:%d" % (vocab_size,num_classes)
    print FLAGS.model_name

    model_dir=os.path.join('temp','nn')
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    with tf.Session() as sess:
        model=getattr(models,FLAGS.model_name)(vocab_size,emb_dim,num_classes)
        saver = tf.train.Saver(tf.global_variables())
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        # sess.run(model.embeddings.assign(pre_embeddings))
        print "Train start!"

        best_dev_f1=0
        best_dev_epoch=0
        no_improve=0
        for epoch in range(FLAGS.max_epoch):
            bar=ProgressBar(max_value=len(train_data)//FLAGS.batch_size+1)
            train_loss=[]
            labels = []
            predicts = []
            for batch_data in bar(utils.minibatches(train_data,FLAGS.batch_size,True)):
                loss,predict=model.train_step(sess,batch_data)
                train_loss.append(loss)
                labels.extend(batch_data[1])
                predicts.extend(predict)
            train_loss=np.mean(train_loss)
            train_f1 = utils.score_all(labels,predicts,tag2id)
            print "Train epoch %d finished. loss:%.3f f1:%.3f" % (epoch,train_loss,train_f1)

            dev_loss=[]
            labels=[]
            predicts=[]
            bar = ProgressBar(max_value=len(train_data) // FLAGS.batch_size+1)
            for batch_data in bar(utils.minibatches(train_data,FLAGS.batch_size,True)):
                loss, predict =model.dev_step(sess,batch_data)
                dev_loss.append(loss)
                labels.extend(batch_data[1])
                predicts.extend(predict)
            dev_loss=np.mean(dev_loss)
            dev_f1 = utils.score_all(labels, predicts,tag2id)
            print "Train epoch %d finished. loss:%.3f f1:%.3f" % (epoch,dev_loss,dev_f1)

            dev_loss=[]
            labels=[]
            predicts=[]
            for batch_data in utils.minibatches(dev_data,FLAGS.batch_size,True):
                loss, predict =model.dev_step(sess,batch_data)
                dev_loss.append(loss)
                labels.extend(batch_data[1])
                predicts.extend(predict)
            dev_loss=np.mean(dev_loss)

            dev_f1 = utils.score_all(labels, predicts,tag2id)
            print "Dev epoch %d finished. loss:%.3f f1:%.3f" % (epoch,dev_loss,dev_f1)

            if dev_f1>best_dev_f1:
                best_dev_f1=dev_f1
                best_dev_epoch=epoch
                no_improve=0
                saver.save(sess, os.path.join(model_dir, FLAGS.model_file))
                print '保存模型!'
            else:
                no_improve+=1
                if no_improve>=5:
                    print "停止训练!"
                    break

            print

        print "Best epoch %d  best f1: %.3f" % (best_dev_epoch,best_dev_f1)


def test():
    x, y = utils.read_file(is_train=True,label_list=['人类作者','自动摘要'])
    x = utils.process(x)
    x = utils.truncation(x)
    word2id,id2word,tag2id,id2tag=utils.build_vocab(x,y,min_df=10)

    x = utils.build_x_ids(x,word2id)
    y = utils.build_y_ids(y,tag2id)

    data=zip(x,y)

    train_data,dev_data =train_test_split(data,test_size=10000,random_state=24)

    vocab_size=len(word2id)
    emb_dim=100
    num_classes=len(tag2id)

    print "训练集数据大小:%d 验证集数据大小:%d" % (len(train_data),len(dev_data))
    print "vocab_size:%d num_classes:%d" % (vocab_size,num_classes)
    print FLAGS.model_name

    model_dir=os.path.join('temp','nn')
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    with tf.Session() as sess:
        model=getattr(models,FLAGS.model_name)(vocab_size,emb_dim,num_classes)
        saver = tf.train.Saver(tf.global_variables())
        model_file = os.path.join('temp', 'nn', FLAGS.model_file)
        saver.restore(sess, model_file)
        print "Restore model from %s" % model_file

        dev_loss=[]
        labels=[]
        predicts=[]
        bar = ProgressBar(max_value=len(dev_data) // FLAGS.batch_size+1)
        for batch_data in bar(utils.minibatches(dev_data,FLAGS.batch_size,True,shuffle=False)):
            loss, predict =model.dev_step(sess,batch_data)
            dev_loss.append(loss)
            labels.extend(batch_data[1])
            predicts.extend(predict)
        dev_loss=np.mean(dev_loss)
        dev_f1 = utils.score_all(labels, predicts,tag2id)
        utils.error_print(predicts,labels,id2tag,zip(*dev_data)[0],id2word)
        print "loss:%.3f f1:%.3f" % (dev_loss,dev_f1)

if __name__ == '__main__':
    if FLAGS.train:
        train()
    else:
        test()


