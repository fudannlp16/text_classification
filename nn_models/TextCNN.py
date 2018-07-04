# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import division
import tensorflow as tf
from Base import Base

class TextCNN(Base):

    def __init__(self,vocab_size,emb_dim,num_classes=4):
        # Hyper parameters
        self.dropout = 0.2
        self.filter_sizes=[1,2,3]
        self.filter_nums=[100,100,100]
        self.init_lr=0.0001
        self.clip_norm=5

        Base.__init__(self,vocab_size,emb_dim)


        temp1 = []
        for i in range(len(self.filter_sizes)):
            conv1=tf.layers.conv1d(inputs=self.emb_inputs,
                                   filters=self.filter_nums[i],
                                   kernel_size=self.filter_sizes[i],
                                   padding='valid',
                                   name='conv1-%d' % i)
            conv1 = tf.nn.relu(conv1)
            pool1=tf.reduce_max(conv1,axis=1)
            temp1.append(pool1)

        self.merge = tf.concat(temp1, axis=1)

        # 训练
        self.train(num_classes)

if __name__ == '__main__':
    TextCNN(100,100)


