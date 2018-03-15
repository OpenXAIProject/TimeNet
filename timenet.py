#Copyright 2018 UNIST under XAI Project supported by Ministry of Science and ICT, Korea

#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at

#   https://www.apache.org/licenses/LICENSE-2.0

#Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import os
os.environ["CUDA_VISIBLE_DEVICES"]="4"
import time
import datetime

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#costomized seq2seq cell
import copy
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import variable_scope


init_time = datetime.datetime.now()

#=======================================Hyperparameter 
learning_rate = 0.001
total_epoch = 1000
batch_size = 100 
dropout = 0.5
hidden_size = 200
index = 0

feed_train = {}
feed_val = {}
outputs = []
one_hots = []
targets = []
DATAS = []
data_size = 50
max_classes = 60

def customized_rnn_seq2seq(encoder_inputs,
                          decoder_inputs,
                          cell,
                          dtype=dtypes.float32,
                          scope=None):
    
    with variable_scope.variable_scope(scope or "basic_rnn_seq2seq"):
        enc_cell = copy.deepcopy(cell)
        encoder_outputs, enc_state = tf.contrib.rnn.static_rnn(enc_cell, encoder_inputs, dtype=dtype)
    return customized_rnn_decoder(encoder_outputs, decoder_inputs, enc_state, cell)

def customized_rnn_decoder(encoder_outputs,
                decoder_inputs,
                initial_state,
                cell,
                loop_function=None,
                scope=None):
    
    with variable_scope.variable_scope(scope or "rnn_decoder"):
        state = initial_state
        outputs = []
        prev = None
        for i, inp in enumerate(decoder_inputs):
            if loop_function is not None and prev is not None:
                with variable_scope.variable_scope("loop_function", reuse=True):
                    inp = loop_function(prev, i)
            if i > 0:
                variable_scope.get_variable_scope().reuse_variables()
            output, state = cell(inp, state)
            outputs.append(output)
            if loop_function is not None:
                prev = output
    return encoder_outputs, outputs, state

def data_maker(dataset):

    datadir = '/home/sohee/UCR_TS_Archive_2015' + '/' + dataset + '/' + dataset
    data_train = np.loadtxt(datadir+'_TRAIN', delimiter=',')
    data_test = np.loadtxt(datadir+'_TEST', delimiter=',')
    DATA = np.concatenate((data_train,data_test),axis=0)

    iter = DATA.shape[0]//data_size
    for i in range(iter):
        CUT_DATA = DATA[50*i:50*(i+1),:]
        DATAS.append(CUT_DATA)
    
def seq2seq_maker(index, DATA):

    X_data = DATA[:,1:]
    y_data = DATA[:,[0]]
    
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.2, random_state=1)
    
    n_feaures = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    
    encoder_input = tf.placeholder(tf.float32, [None, X_train.shape[1]]) 
    decoder_input = tf.placeholder(tf.float32, [None, X_train.shape[1]])
    target = tf.placeholder(tf.int64, [None, 1])
    target_one_hot = tf.one_hot(target, max_classes)
    one_hot = tf.reshape(target_one_hot, [-1, max_classes])
    
    targets.append(target)
    one_hots.append(one_hot)
    
    #logits_size=[40,100] labels_size=[480,1]
    
    
    with tf.variable_scope("rnn_"+str(index)):
        cell = tf.contrib.rnn.GRUCell(num_units=hidden_size)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.5)
        en_outputs, de_outputs, state = customized_rnn_seq2seq([encoder_input], [decoder_input], cell)
        de_outputs = tf.reshape(de_outputs, [-1, hidden_size]) #3D -> 2D
        outputs.append(de_outputs)
    
    feed_train[encoder_input] = X_train
    feed_train[decoder_input] = X_train
    feed_train[target] = y_train
    
    feed_val[encoder_input] = X_val
    feed_val[decoder_input] = X_val
    feed_val[target] = y_val


traindatasets = ['Plane', 'Gun_Point', 'ArrowHead', 'WordsSynonyms', 'ToeSegmentation1', 'FISH', 'ShapeletSim', 'ShapesAll', 'SonyAIBORobotSurfaceII',
             'Lighting7', 'ToeSegmentation2', 'DiatomSizeReduction', 'Ham', 'SonyAIBORobotSurface', 'TwoLeadECG', 'FacesUCR']

for dataset in traindatasets:
    data_maker(dataset)    

for index, DATA in enumerate(DATAS):
    seq2seq_maker(index, DATA)

W = tf.Variable(tf.random_normal([hidden_size, max_classes]), name="W") 
b = tf.Variable(tf.random_normal([max_classes]), name="b")
logits = [tf.matmul(output, W) + b for output in outputs]
with tf.variable_scope("cost"):
    loss = []
    for logit, one_hot in zip(logits, one_hots):
        loss.append(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=one_hot))
    cost= tf.reduce_mean(loss)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)  

saver = tf.train.Saver()

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    
    print(sess.run("W:0"))
    print(sess.run("b:0"))
    print("============================sess init")
    start_time = time.time()
    train_losses = []
    valid_losses = []
    for epoch in range(total_epoch):
        _, tr_loss = sess.run([optimizer, cost], feed_dict = feed_train)
        _, va_loss = sess.run([optimizer, cost], feed_dict = feed_val)
        train_losses += [tr_loss]
        valid_losses += [va_loss]
        if epoch % 100 == 0:  
            print("Epoch {}/{} took {:.3f}s".format(epoch + 1, total_epoch,time.time() - start_time))
            print("  Train      loss : %.6f"%(train_losses[epoch]))
            print("  Validation loss : %.6f"%(valid_losses[epoch]))
    print("It took", time.time() - start_time, "seconds to train for", total_epoch, "epochs.")        
    print("============================ training end")
    
    #loss  그래프를 확인한다. 
    plt.plot(train_losses, '-b', label='Train loss')
    plt.plot(valid_losses, '-r', label='Valid loss')
    plt.legend(loc=0)
    plt.title('Loss graph')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
        
    print(sess.run("W:0"))
    print(sess.run("b:0"))
    now = datetime.datetime.now().strftime("%Y-%m-%d-%Hh-%Mm-%Ss")
    save_path = "./CheckPoint/"+now+".ckpt"
    saver.save(sess, save_path)
    print("model saved to ", save_path)

tf.reset_default_graph()
saver = tf.train.import_meta_graph(save_path+".meta")
print("... meta graph loaded")

class Timenet:
    
    def __init__(self, sess, dataset):
        print("===============================================================  ", dataset)
        self.sess = sess
        self.dataset = dataset
        self.model(self.dataset)
        
    def model(self, dataset):
        
        datadir = '/home/sohee/UCR_TS_Archive_2015' + '/' + dataset + '/' + dataset
        data_train = np.loadtxt(datadir+'_TRAIN', delimiter=',')
        data_test = np.loadtxt(datadir+'_TEST', delimiter=',')

        #train data
        self.X_train = data_train[:,1:]
        self.y_train = data_train[:,[0]]

        #test data
        self.X_test = data_test[:,1:]
        self.y_test = data_test[:,[0]]

        n_features = self.X_train.shape[1]
        n_classes = len(np.unique(self.y_train))
        # placeholder
        self.encoder = tf.placeholder(tf.float32, [None, n_features], name="encoder") 
        self.decoder = tf.placeholder(tf.float32, [None, n_features], name="decoder")
        self.targets = tf.placeholder(tf.int64, [None, 1], name="targets")
        one_hot = tf.one_hot(self.targets, max_classes)
        one_hot = tf.reshape(one_hot, [-1, max_classes])

        with tf.variable_scope("seq2seq"+dataset):  
            cell = tf.contrib.rnn.GRUCell(num_units=hidden_size)
            encoder_outputs, outputs, states = customized_rnn_seq2seq([self.encoder], [self.decoder], cell)
            outputs = tf.reshape(outputs, [-1, hidden_size]) #3D -> 2D # output 모양이 항상 [? , hidden_size]으로 고정됨
            
            W = tf.get_default_graph().get_tensor_by_name("W:0")
            b = tf.get_default_graph().get_tensor_by_name("b:0")
            logits = tf.matmul(outputs, W) + b
            hypothesis = tf.nn.softmax(logits)

        with tf.variable_scope("cost"+dataset):
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot))
            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost) 

        with tf.variable_scope("eval"+dataset):
            self.prediction = tf.argmax(hypothesis, 1) 
            self.true = tf.argmax(one_hot, 1)
            self.correct_prediction = tf.equal(self.prediction, self.true)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction,tf.float32))
           
    def optimize_cost(self):
        return self.sess.run([self.optimizer, self.cost], feed_dict={ self.encoder : self.X_train, 
                                                    self.decoder : self.X_train, 
                                                    self.targets : self.y_train })
    
    
    def print_accuracy(self):
        pred = self.sess.run(self.prediction, feed_dict={ self.encoder : self.X_test, 
                                                          self.decoder : self.X_test})
        answer = self.sess.run(self.true, feed_dict={ self.targets : self.y_test})
        correct_prediction = self.sess.run(tf.equal(pred, answer))
        accuracy = self.sess.run(tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))
        return pred, answer, accuracy

testdatasets = ['synthetic_control','PhalangesOutlinesCorrect', 'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW', 
                'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect', 'MiddlePhalanxTW', 'ProximalPhalanxOutlineAgeGroup', 
                'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW', 'ElectricDevices', 'MedicalImages', 'SwedishLeaf', 'Two_Patterns', 'ECG5000', 
                'ECGFiveDays', 'ChlorineConcentration', 'Adiac', 'Strawberry', 'Cricket_X', 'Cricket_Y', 'Cricket_Z', 'uWaveGestureLibrary_X', 
                'uWaveGestureLibrary_Y', 'uWaveGestureLibrary_Z', 'yoga', 'wafer', 'FordA', 'FordB']
minitest =  [ 'yoga', 'wafer', 'FordA', 'FordB' ]

for dataset in testdatasets :   
    sess = tf.Session()
    m = Timenet(sess, dataset)
    
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, save_path)
   
    losses = []
    start_time = time.time()
    for epoch in range(total_epoch):
        _, cost = m.optimize_cost()
        losses += [cost] 
        
        if epoch == 0:  
            print("Epoch {}/1000 ".format(epoch+1), "cost : ", cost)
        if epoch == (total_epoch//2-1) :  
            print("Epoch {}/1000 ".format(epoch+1), "cost : ", cost)
        if epoch == (total_epoch-1) :  
            print("Epoch {}/1000 ".format(epoch+1), "cost : ", cost)         
    print(" It took ", time.time()-start_time, "s.")
    
    plt.plot(losses, '-b', label='Train loss')
    plt.title(dataset)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
    
    pred, answer, accuracy = m.print_accuracy()
    print("")
    print(dataset, ', Accuracy:', accuracy, ', Error rate : ', 1-accuracy)
    print("")
    
    for p, a in zip(pred, answer):
        print("predict : {}, answer : {} = > [{}]".format(p, a, p==a))

finish_time = datetime.datetime.now()
print("start   : ", init_time.strftime("%Y-%m-%d-%Hh-%Mm"))
print("finish  : ", finish_time.strftime("%Y-%m-%d-%Hh-%Mm"))
print("Total Running Time : {}".format(finish_time - init_time))