
import sys
import warnings

from PyEMD import CEEMDAN
import numpy
import numpy as np
import pandas as pd
import math
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Input, Bidirectional
from sklearn import metrics
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
from keras.layers import Dense, Dropout,RepeatVector,Flatten, Conv1D,MaxPooling1D
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers import Layer
import keras.backend as K
from vmdpy import VMD

import time
# import dataframe_image as dfi

import warnings
warnings.filterwarnings("ignore")

numpy.random.seed(1234)
import tensorflow as tf
tf.random.set_seed(1234)

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-2):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back+2, 0])
    return np.array(dataX), np.array(dataY)

def percentage_error(actual, predicted):
    res = numpy.empty(actual.shape)
    for j in range(actual.shape[0]):
        if actual[j] != 0:
            res[j] = (actual[j] - predicted[j]) / actual[j]
        else:
            res[j] = predicted[j] / np.mean(actual)
    return res

def modified_huber_loss(delta=1.0):
    def loss(y_true, y_pred):
        error = y_true - y_pred
        tau = K.exp(-K.abs(error))
        is_small_error = (K.abs(error) <= delta)
        small_error_loss = 0.5 * K.square(error)
        large_error_loss = delta * (K.abs(error) - 0.5 * delta)

        weighted_loss = tf.where(is_small_error, tau * small_error_loss, (1 - tau) * large_error_loss)

        return K.mean(weighted_loss)

    return loss

class attention(Layer):
    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)

    def build(self,input_shape):
        self.W=self.add_weight(name="att_weight",shape=(input_shape[-1],1),initializer="normal")
        self.b=self.add_weight(name="att_bias",shape=(input_shape[1],1),initializer="zeros")        
        super(attention, self).build(input_shape)

    def call(self,x):
        et=K.squeeze(K.tanh(K.dot(x,self.W)+self.b),axis=-1) #score
        at=K.softmax(et)  #attention weights
        at=K.expand_dims(at,axis=-1)
        output=x*at  
        return K.sum(output,axis=1) # context vector

    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[-1])

    def get_config(self):
        return super(attention,self).get_config()

def mean_absolute_percentage_error(y_true, y_pred): 
    return numpy.mean(numpy.abs(percentage_error(numpy.asarray(y_true), numpy.asarray(y_pred)))) * 100


def main():
    #reading the dataset
    import pandas as pd
    df=pd.read_excel(r"UP08.xlsx",index_col='Date_Time')
    df=df['2017-03-31 20:00:00':]
    df=df[['dissolvedoxygenmeasured']]
    
    start_time = time.time()

    df1 = df
    df1 = df1.reset_index()

    df2 = df1['dissolvedoxygenmeasured']
    a = np.array(df2)

    emd = CEEMDAN(epsilon=0.05)
    emd.noise_seed(12345)

    IMFs = emd(a)

    full_imf=pd.DataFrame(IMFs)
    data_imf=full_imf.T

    CEEMDAN_time=time.time() - start_time
   
    pred_test=[]
    test_ori=[]
    pred_train=[]
    train_ori=[]

    start_time = time.time()
    
    for i in range(1):
        alpha = 2000       # moderate bandwidth constraint  
        tau = 0.            # noise-tolerance (no strict fidelity enforcement)  
        K = 10              # 10 modes  
        DC = 0             # no DC part imposed  
        init = 1           # initialize omegas uniformly  
        tol = 1e-7  

        # Run VMD code  
        u, u_hat, omega = VMD(IMFs[0], alpha, tau, K, DC, init, tol) 
        
        for i in range(K): 
            datasetss2=pd.DataFrame(u[i])
            datasets=datasetss2.values
            X, Y = create_dataset(datasets, 5)
            train_size = int(len(X) * 0.75)
            test_size = len(X) - train_size
            trainX, testX = X[0:train_size], X[train_size:len(X)]
            trainY, testY = Y[0:train_size], Y[train_size:len(Y)]

            X_train=pd.DataFrame(trainX)
            Y_train=pd.DataFrame(trainY)
            X_test=pd.DataFrame(testX)
            Y_test=pd.DataFrame(testY)
            sc_X = MinMaxScaler()
            sc_y = MinMaxScaler()
            X= sc_X.fit_transform(X_train)
            y= sc_y.fit_transform(Y_train)
            X1= sc_X.transform(X_test)
            y1= sc_y.transform(Y_test)
            y=y.ravel()
            y1=y1.ravel()  

            numpy.random.seed(1234)
            import tensorflow as tf
            tf.random.set_seed(1234)

            from keras.models import Sequential
            from keras.layers.core import Dense, Dropout, Activation
            from keras.layers import LSTM
            
            trainX = X.reshape((X.shape[0], 1, X.shape[1]))
            testX = X1.reshape((X1.shape[0], 1, X1.shape[1]))

            model = Sequential()
            model.add(Bidirectional(LSTM(units = 5, activation="relu", return_sequences=True, input_shape=(trainX.shape[1], trainX.shape[2]))))
            model.add(attention())
            model.add(RepeatVector(X_train.shape[1], name="bottleneck_output"))
            model.add(Flatten())
            model.add(Dense(2))
            model.add(Dense(1))

            opt = keras.optimizers.Adam(learning_rate=0.01)
            model.compile(loss=modified_huber_loss(),optimizer=opt)

            es = tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=10,verbose=1,mode="min")
            model.fit(trainX, y, validation_split = 0.1, epochs = 1000, batch_size = 64,verbose=1, callbacks=[es]) 

            # make predictions
            y_pred_train = model.predict(trainX)
            y_pred_test = model.predict(testX)

            y_pred_test= numpy.array(y_pred_test).ravel()
            y_pred_test=pd.DataFrame(y_pred_test)

            y1=pd.DataFrame(y1)
            y=pd.DataFrame(y)                           

            y_pred_train= numpy.array(y_pred_train).ravel()
            y_pred_train=pd.DataFrame(y_pred_train)

            y_test= sc_y.inverse_transform (y1)
            y_train= sc_y.inverse_transform (y)
            y_pred_test1= sc_y.inverse_transform (y_pred_test)
            y_pred_train1= sc_y.inverse_transform (y_pred_train)

            pred_test.append(y_pred_test1)
            test_ori.append(y_test)
            pred_train.append(y_pred_train1)
            train_ori.append(y_train)    

    for i in range(1,data_imf.shape[1]):

        datasetss2=pd.DataFrame(data_imf[i])
        datasets=datasetss2.values
        X, Y = create_dataset(datasets, 5)
        train_size = int(len(X) * 0.75)
        test_size = len(X) - train_size
        trainX, testX = X[0:train_size], X[train_size:len(X)]
        trainY, testY = Y[0:train_size], Y[train_size:len(Y)]

        X_train=pd.DataFrame(trainX)
        Y_train=pd.DataFrame(trainY)
        X_test=pd.DataFrame(testX)
        Y_test=pd.DataFrame(testY)
        sc_X = MinMaxScaler()
        sc_y = MinMaxScaler()
        X= sc_X.fit_transform(X_train)
        y= sc_y.fit_transform(Y_train)
        X1= sc_X.transform(X_test)
        y1= sc_y.transform(Y_test)
        y=y.ravel()
        y1=y1.ravel()  

        numpy.random.seed(1234)
        import tensorflow as tf
        tf.random.set_seed(1234)

        from keras.models import Sequential
        from keras.layers.core import Dense, Dropout, Activation
        from keras.layers import LSTM
      
        trainX = X.reshape((X.shape[0], 1, X.shape[1]))
        testX = X1.reshape((X1.shape[0], 1, X1.shape[1]))

        model = Sequential()
        model.add(Bidirectional(LSTM(units = 5, activation="relu", return_sequences=True, input_shape=(trainX.shape[1], trainX.shape[2]))))
        model.add(attention())
        model.add(RepeatVector(X_train.shape[1], name="bottleneck_output"))
        model.add(Flatten())
        model.add(Dense(2))
        model.add(Dense(1))

        opt = keras.optimizers.Adam(learning_rate=0.01)
        model.compile(loss=modified_huber_loss(),optimizer=opt)

        es = tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=10,verbose=1,mode="min")
        model.fit(trainX, y, validation_split = 0.1, epochs = 1000, batch_size = 64,verbose=1, callbacks=[es]) 

        # make predictions
        y_pred_train = model.predict(trainX)
        y_pred_test = model.predict(testX)

        y_pred_test= numpy.array(y_pred_test).ravel()
        y_pred_test=pd.DataFrame(y_pred_test)

        y1=pd.DataFrame(y1)
        y=pd.DataFrame(y)                           

        y_pred_train= numpy.array(y_pred_train).ravel()
        y_pred_train=pd.DataFrame(y_pred_train)

        y_test= sc_y.inverse_transform (y1)
        y_train= sc_y.inverse_transform (y)
        y_pred_test1= sc_y.inverse_transform (y_pred_test)
        y_pred_train1= sc_y.inverse_transform (y_pred_train)

        pred_test.append(y_pred_test1)
        test_ori.append(y_test)
        pred_train.append(y_pred_train1)
        train_ori.append(y_train)                          

    result_pred_test= pd.DataFrame.from_records(pred_test)
    result_pred_train= pd.DataFrame.from_records(pred_train)

    a=result_pred_test.sum(axis = 0, skipna = True) 
    b=result_pred_train.sum(axis = 0, skipna = True) 

    data_test = np.array(pred_test)
    result_pred_test= pd.DataFrame.from_records(data_test)
    a=result_pred_test.sum(axis = 0, skipna = True) 

    data_train = np.array(pred_train)
    result_pred_train= pd.DataFrame.from_records(data_train)
    b=result_pred_train.sum(axis = 0, skipna = True) 

    datasets=df.values
    X, Y = create_dataset(datasets, 5)

    train_size = int(len(X) * 0.75)
    test_size = len(X) - train_size
    trainX, testX = X[0:train_size], X[train_size:len(X)]
    trainY, testY = Y[0:train_size], Y[train_size:len(Y)]

    X_train=pd.DataFrame(trainX)
    Y_train=pd.DataFrame(trainY)
    X_test=pd.DataFrame(testX)
    Y_test=pd.DataFrame(testY)

    sc_X = MinMaxScaler()
    sc_y = MinMaxScaler() 
    X= sc_X.fit_transform(X_train)
    y= sc_y.fit_transform(Y_train)
    X1= sc_X.transform(X_test)
    y1= sc_y.transform(Y_test)
    y=y.ravel()
    y1=y1.ravel()

    trainX = X.reshape((X.shape[0], 1, X.shape[1]))
    testX = X1.reshape((X1.shape[0], 1, X1.shape[1]))

    numpy.random.seed(1234)
    import tensorflow as tf
    tf.random.set_seed(1234)

    y1=pd.DataFrame(y1)
    y=pd.DataFrame(y)
    y_test= sc_y.inverse_transform (y1)
    y_train= sc_y.inverse_transform (y)

    a = pd.DataFrame(a)    
    y_test = pd.DataFrame(y_test)    
    b = pd.DataFrame(b)    
    y_train = pd.DataFrame(y_train)                                 

    #summarize the fit of the model on train data
    mape_train = mean_absolute_percentage_error(trainY,b)
    rmse_train = sqrt(mean_squared_error(trainY,b))
    mae_train = metrics.mean_absolute_error(trainY,b)
    r2_train = r2_score(trainY,b)

    # summarize the fit of the model on test data
    mape_test = mean_absolute_percentage_error(testY,a)
    rmse_test = sqrt(mean_squared_error(testY,a))
    mae_test = metrics.mean_absolute_error(testY,a)
    r2_test = r2_score(testY,a)

    # train scores
    print("The metrics for the training data are: ")
    print(mape_train)
    print(rmse_train)
    print(mae_train)
    print(r2_train)

    # test scores
    print("The metrics for the testing data are: ")
    print(mape_test)
    print(rmse_test)
    print(mae_test)
    print(r2_test)

    b.to_csv("UP08_CV_biLSTM_Attention_Predicted_Train.csv")
    a.to_csv("UP08_CV_biLSTM_Attention_Predicted_Test.csv")

    LSTM_time=time.time() - start_time
    print("--- %s seconds -#  UP08_CV_biLSTM_Attention---" % (CEEMDAN_time +LSTM_time))

    a1 = pd.DataFrame([mape_train , rmse_train,mae_train, r2_train])
    a2 = pd.DataFrame([mape_test , rmse_test , mae_test , r2_test])
    d1 = pd.concat([a1, a2], axis = 1)
    d1.columns = ["Train", 'Test']
    d1 = d1.set_axis(['MAPE(%)','RMSE','MAE','r2'], axis='index')
    index = d1.index
    index.name = "Results"
    pd.DataFrame(d1).to_csv("UP08_CV_biLSTM_Attention.csv")

main()
