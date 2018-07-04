# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import division
import tensorflow as tf
from tensorflow.contrib import rnn
from Base import Base

class TextRNN(Base):

    def __init__(self,vocab_size,emb_dim,num_classes=4):
        # Hyper parameters
        self.dropout = 0.4
        self.num_unit=100
        self.num_layer=1
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

        self.merge = tf.concat([state_fw[-1],state_bw[-1]],axis=-1)

        # 训练
        self.train(num_classes)

if __name__ == '__main__':
    TextRNN(100,100)


