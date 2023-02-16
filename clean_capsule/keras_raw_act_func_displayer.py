#

import matplotlib.pyplot as plt
import numpy
from tensorflow import keras
import tensorflow
from tensorflow.keras import layers
from keras.models import Sequential
from numpy import loadtxt
from numpy import loadtxt
from keras.layers import Dense
import numpy as np
import keras as K
import math
import random
import os

print(keras.__version__)
print(numpy.__version__)
print(tensorflow.__version__)


def setup_dir():
    isExist = os.path.exists('results')
    if isExist !=True:
        os.mkdir('results')
    isExist = os.path.exists('results/layer 1')
    if isExist !=True:
        os.mkdir('results/layer 1')
    
    isExist = os.path.exists('results/layer 2')
    if isExist !=True:
        os.mkdir('results/layer 2')
    isExist = os.path.exists('results/layer 3')
    if isExist !=True:
        os.mkdir('results/layer 3')


def plts(layer_,name_fig):
    plt.xlabel("Neuron number ")
    plt.ylabel("Value of output of neuron")
    plt.title(name_fig)
    plt.plot(layer_)
    plt.savefig(name_fig) 
    plt.close()



class act_v_caller:

    def __init__(self):
        self._weights_saver =None
        self.trainset_input = []
        self.trainset_output =[]
        self.evalulation_set =[]
        self.out_put_softmaxs =[]



    def gens(self, cut_=40, begin_=1, end_=10):
        randomlist = []
        for i in range(0,cut_):
            n = random.randint(begin_,end_)
            randomlist.append(n)
        return randomlist

    def generate_train_set(self):
        #ge sets
        
        for i in range(0, 55):
            self.trainset_input.append(self.gens())
        
        
        
        z = self.gens(55,0,4)
        for i in z:
            x = [0]*5
            x[i]=1
            self.trainset_output.append(x)
    
  
        
    def generate_eval_set(self):
        #ge sets
       
        for i in range(0, 10):
            self.evalulation_set.append(self.gens())


    def gen_model_with_weights(self):
        #gen mocel
        model = Sequential()
        model.add(	Dense(60, input_dim=40, activation='relu'))
        model.add(	Dense(80, activation='relu'))
        model.add(	Dense(100, activation='relu'))	
        model.add(	Dense(75, activation='relu'))	
        model.add(	Dense(50, activation='relu'))
        model.add(  Dense(25, activation='relu'))
        model.add(	Dense(5, activation='softmax'))
        model.summary()
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.fit(self.trainset_input, self.trainset_output, epochs=100, verbose=1)
        self._weights_saver=model.get_weights()

    def geen_test(self, list_func=['relu','tanh','sigmoid','LeakyReLU','elu','selu','exponential','softsign', 'softplus', 'softmax', 'linear']):
        print(list_func)
        for func_ in list_func:
            model2 = Sequential()
            
            model2.add(	Dense(60, input_dim=40, activation='relu'))
            model2.add(	Dense(80, activation='relu'))
            model2.add(	Dense(100, activation=func_))	
            model2.add(	Dense(75, activation='relu'))	
            model2.add(	Dense(50, activation='relu'))
            model2.add(  Dense(25, activation='relu'))
            model2.add(	Dense(5, activation='softmax'))
            #model2.summary()

            model2.set_weights(self._weights_saver)
            model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

            #puts extra models here, bcz i am so lazy to find difrent way for implementations, lol shagy 

            model3 = Sequential()
            model3.add(	Dense(60, weights=model2.layers[0].get_weights(), input_dim=40, activation='relu'))
            model3.add(	Dense(80, weights=model2.layers[1].get_weights(), activation='relu'))


            model4 = Sequential()
            model4.add(	Dense(60, weights=model2.layers[0].get_weights(), input_dim=40, activation='relu'))
            model4.add(	Dense(80, weights=model2.layers[1].get_weights(), activation='relu'))
            model4.add(	Dense(100, weights=model2.layers[2].get_weights(), activation=func_))	
     

            model5 = Sequential()
            model5.add(	Dense(60, weights=model2.layers[0].get_weights(), input_dim=40, activation='relu'))
            model5.add(	Dense(80, weights=model2.layers[1].get_weights(), activation='relu'))
            model5.add(	Dense(100, weights=model2.layers[2].get_weights(), activation=func_))	
            model5.add(	Dense(75, weights=model2.layers[3].get_weights(), activation='relu'))	

            for example in range( len(self.evalulation_set) ):
                cl=model2.predict([self.evalulation_set[example]])
                cl=cl[0]
                cl=np.round(cl)
                print("function is: ",func_, "example numb: ", example, "prediction is: ", cl )
                if example==4 and func_=='relu':
                   
                    out1=model3.predict([self.evalulation_set[example]])
                    name=("output values for act fun: ",func_, "for layer with lenght 80: ", out1[0].tolist() )
                    name2="results/layer 1/output values for act fun "+str(func_)+" for layer with lenght 80 "
                    plts(out1[0].tolist(),name2)
                    print(name)
                if example ==4: 
                    
                    out2=model4.predict([self.evalulation_set[example]])
                    name=("output values for act fun: ",func_, "for layer with lenght 100: ", out2[0].tolist() )
                    name2="results/layer 2/output values for act fun "+str(func_)+ " for layer with lenght 100 "
                    print(name)
                    plts(out2[0].tolist(),name2)

                    out3=model5.predict([self.evalulation_set[example]])
                    name=("output values for act fun: ",func_, "for layer with lenght 75: ", out3[0].tolist() )
                    name2="results/layer 3/output values for act fun "+str(func_)+ " for layer with lenght 75 "
                    print(name)
                    plts(out3[0].tolist(),name2)
                    

setup_dir()
test= act_v_caller()
test.generate_train_set()
test.generate_eval_set()
test.gen_model_with_weights()
test.geen_test()


