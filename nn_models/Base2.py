# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import division
import tensorflow as tf
import numpy as np

class Base2(object):
    """
    父类,提供各个nn模型通用方法(Hierarchical)
    """
    def __init__(self,vocab_size,emb_dim):
        self.build_inputs(vocab_size,emb_dim)

    # 模型通用输入模块
    def build_inputs(self,vocab_size,emb_dim):
        self.x = tf.placeholder(dtype=tf.int32, shape=[None, None], name='x')
        self.y = tf.placeholder(dtype=tf.int32, shape=[None], name='y')
        self.dropout_keep = tf.placeholder(dtype=tf.float32, name='dropout_keep')

        self.lengths = tf.reduce_sum(tf.cast(self.x > tf.zeros_like(self.x), tf.int32), axis=-1,name='length')

        self.embeddings = tf.get_variable(name='embeddings',shape=[vocab_size, emb_dim],dtype=tf.float32)
        self.emb_inputs = tf.nn.embedding_lookup(self.embeddings, self.x)
        self.emb_inputs = tf.nn.dropout(self.emb_inputs, self.dropout_keep)

        self.emb_inputs = tf.reshape(self.emb_inputs,[-1,20,20,emb_dim])

    def train(self,num_classes):
        f1 = tf.layers.dense(self.merge,100,activation=tf.nn.tanh)
        f1 = tf.layers.batch_normalization(f1)
        f1 = tf.nn.dropout(f1, self.dropout_keep)

        f2 = tf.layers.dense(f1, num_classes)

        self.logits = f2
        self.probs=tf.nn.softmax(self.logits,dim=1)
        self.predict=tf.argmax(self.logits,axis=1)
        self.loss=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y,logits=self.logits)
        self.loss=tf.reduce_mean(self.loss)

        # 优化
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            tvars = tf.trainable_variables()
            clipped_gradients, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), clip_norm=self.clip_norm)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.init_lr)
            self.train_op = optimizer.apply_gradients(zip(clipped_gradients, tvars))

    def train_step(self,sess,batch_data):
        # 训练
        x,y = batch_data
        x=np.array(x)
        batch,sequence_length=x.shape
        if sequence_length<400:
            x=np.concatenate([x,np.zeros(batch,400-sequence_length)],axis=1)
        feed_dict={
            self.x:x,
            self.y:y,
            self.dropout_keep:1-self.dropout,
        }
        _,loss,predict= sess.run([self.train_op,self.loss,self.predict],feed_dict)
        predict=predict.tolist() # 转化为list
        return loss,predict

    def dev_step(self,sess,batch_data):
        # 验证
        x,y = batch_data
        x = np.array(x)
        batch, sequence_length = x.shape
        if sequence_length < 400:
            x = np.concatenate([x, np.zeros(batch, 400 - sequence_length)], axis=1)
        feed_dict={
            self.x:x,
            self.y:y,
            self.dropout_keep:1.0,
        }
        loss,predict = sess.run([self.loss,self.predict],feed_dict)
        predict = predict.tolist()  # 转化为list
        return loss,predict

    def eval_step(self,sess,batch_data):
        # 预测
        x = batch_data
        x = np.array(x)
        batch, sequence_length = x.shape
        if sequence_length < 400:
            x = np.concatenate([x, np.zeros(batch, 400 - sequence_length)], axis=1)
        feed_dict={
            self.x:x,
            self.dropout_keep:1.0,
        }
        probs =sess.run(self.probs,feed_dict)
        probs = probs.tolist()  # 转化为list
        return probs

if __name__ == '__main__':
    pass
