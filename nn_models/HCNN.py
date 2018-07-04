# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import division
import tensorflow as tf
from Base2 import Base2

class HCNN(Base2):
    """Hierarchical CNN for text classification"""

    def __init__(self,vocab_size,emb_dim,num_classes=4):
        # Hyper parameters
        self.dropout = 0.2
        self.filter_sizes=[1,2,3,4]
        self.filter_nums=[64,64,64,64]
        self.filter_sizes2 = [1,2,3,4]
        self.filter_nums2 = [64,64,64,64]
        self.init_lr=0.001
        self.clip_norm=5

        Base2.__init__(self,vocab_size,emb_dim)

        # 单词级别卷积
        word_inputs=tf.reshape(self.emb_inputs,[-1,20,emb_dim])
        temp1 = []
        for i in range(len(self.filter_sizes)):
            conv1=tf.layers.conv1d(inputs=word_inputs,
                                   filters=self.filter_nums[i],
                                   kernel_size=self.filter_sizes[i],
                                   padding='valid',
                                   name='conv1-%d' % i)
            conv1 = tf.nn.relu(conv1)
            poop1=tf.reduce_max(conv1,axis=1)
            temp1.append(poop1)

        # 句子级别卷积
        sent_inputs = tf.reshape(tf.concat(temp1, axis=1),[-1,20,64*4])
        temp2 = []
        for i in range(len(self.filter_sizes2)):
            conv2=tf.layers.conv1d(inputs=sent_inputs,
                                   filters=self.filter_nums2[i],
                                   kernel_size=self.filter_sizes2[i],
                                   padding='valid',
                                   name='conv2-%d' % i)
            conv2 = tf.nn.relu(conv2)
            poop2=tf.reduce_max(conv2,axis=1)
            temp2.append(poop2)

        self.merge = tf.concat(temp2, axis=1)
        # 训练
        self.train(num_classes)

if __name__ == '__main__':
    HierarchyCNN(100,100)