# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import division
import tensorflow as tf
from tensorflow.contrib import rnn
from Base import Base

class RCNN(Base):
    """ Recurrent CNN for text classification"""

    def __init__(self,vocab_size,emb_dim,num_classes=4):
        # Hyper parameters
        self.dropout = 0.4
        self.num_unit=100
        self.num_layer=1
        self.filter_sizes = [1, 2, 3]
        self.filter_nums = [64, 64, 64]
        self.init_lr=0.001
        self.clip_norm=5

        Base.__init__(self,vocab_size,emb_dim)

        def cell():
            cell=rnn.DropoutWrapper(rnn.GRUCell(num_units=self.num_unit))
            return cell
        cell_fw=[cell() for _  in range(self.num_layer)]
        cell_bw =[cell() for _ in range(self.num_layer)]
        outputs,state_fw,state_bw = rnn.stack_bidirectional_dynamic_rnn(
            cells_fw=cell_fw,
            cells_bw=cell_bw,
            inputs=self.emb_inputs,
            sequence_length=self.lengths,
            dtype=tf.float32
        )

        outputs=tf.concat([outputs,self.emb_inputs],axis=-1)

        temp1 = []
        for i in range(len(self.filter_sizes)):
            conv1=tf.layers.conv1d(inputs=outputs,
                                   filters=self.filter_nums[i],
                                   kernel_size=self.filter_sizes[i],
                                   padding='valid',
                                   name='conv1-%d' % i)
            conv1 = tf.nn.relu(conv1)
            conv1=tf.layers.batch_normalization(inputs=conv1)
            poop1=tf.reduce_max(conv1,axis=1)
            temp1.append(poop1)

        self.merge = tf.concat(temp1, axis=1)
        # 训练
        self.train(num_classes)

if __name__ == '__main__':
    RnnCNN(100,100)