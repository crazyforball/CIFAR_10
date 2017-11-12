'''
Created on May 22, 2017

@author: yang
'''
import numpy as np
from time import time
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from scipy import io
import scipy
from sklearn.model_selection import KFold

# This is the function to unpickle raw data
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# This is the function to plot confusion matrix
def plot_confusion_matrix(cm, classes, i):
    df_cm = pd.DataFrame(cm, index = [idx for idx in classes],
                  columns = [idx for idx in classes])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True, fmt='g')
    plt.xticks(fontsize=8, rotation='vertical')
    plt.yticks(fontsize=8, rotation='horizontal')
    plt.tight_layout()
    plt.title('Confusion Matrix')
    #plt.show()
    plt.savefig('confusion_matrix_'+str(i)+'.png', format='png')


# Get data metrix
raw_dataset_path = "../dataset/cifar-10-batches-py/"

raw_data_1_dict_path = raw_dataset_path + "data_batch_1"
raw_data_1_dict = unpickle(raw_data_1_dict_path)
raw_data_1 = raw_data_1_dict.get(b'data')

raw_data_2_dict_path = raw_dataset_path + "data_batch_2"
raw_data_2_dict = unpickle(raw_data_2_dict_path)
raw_data_2 = raw_data_2_dict.get(b'data')

raw_data_3_dict_path = raw_dataset_path + "data_batch_3"
raw_data_3_dict = unpickle(raw_data_3_dict_path)
raw_data_3 = raw_data_3_dict.get(b'data')

raw_data_4_dict_path = raw_dataset_path + "data_batch_4"
raw_data_4_dict = unpickle(raw_data_4_dict_path)
raw_data_4 = raw_data_4_dict.get(b'data')

raw_data_5_dict_path = raw_dataset_path + "data_batch_5"
raw_data_5_dict = unpickle(raw_data_5_dict_path)
raw_data_5 = raw_data_5_dict.get(b'data')

X = np.vstack((raw_data_1,raw_data_2,raw_data_3,raw_data_4,raw_data_5))

# Get labels.
labels_1 = raw_data_1_dict.get(b'labels')
labels_2 = raw_data_2_dict.get(b'labels')
labels_3 = raw_data_3_dict.get(b'labels')
labels_4 = raw_data_4_dict.get(b'labels')
labels_5 = raw_data_5_dict.get(b'labels')

y = np.hstack((labels_1,labels_2,labels_3,labels_4,labels_5))

# Get label names.
# label_names_dict_path = "cifar-10-batches-py/batches.meta"
# label_names_dict = unpickle(label_names_dict_path)
# label_names = np.array(label_names_dict.get(b'label_names'))
label_names = np.array(['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'])

# normalize data
data_normalized = X / 255.0


###############################################################################
# The CNN classifier
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils, generic_utils
import keras

# set parameters
batch_size = 50
num_classes = 10
epochs = 60

# transform data
data = np.reshape(data_normalized,(50000,32,32,3))
label = np_utils.to_categorical(y, num_classes)

# set output file path
result_file = 'cnn_report.csv'
output = open(result_file, 'a')

# set the proportion of training data and validation data
kf = KFold(n_splits=10)

# set a variable to record the number of cross-validation performed
rd = 0

# set a variable to record average confusion matrix
cm_avg = np.zeros((num_classes,num_classes))

print("start")

# train classifier for 10-fold cross-validation
for train, valid in kf.split(data):
    
    rd += 1
    
    t0=time()
    
    #create model and add layers 
    model = Sequential()
  
    model.add(Conv2D(32, (5, 5), padding='same',
                     input_shape=data.shape[1:]))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
     
    model.add(Conv2D(32, (5, 5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
      
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
     
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('tanh'))
     
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
     
# initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    
# compile model
    model.compile(loss='categorical_crossentropy', optimizer=opt,class_mode="categorical", metrics=['accuracy'])

# split data into training data and validation data
    X_train, X_valid, y_train, y_valid = data[train], data[valid], label[train], label[valid]
    
# Fit model to data
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
           validation_data=(X_valid, y_valid), shuffle=True)
    
# predict validation data
    actual = y[valid]
    predicted = model.predict_classes(X_valid, batch_size=batch_size, verbose=1)
    
    t1 = time() 
    
# output results
    try:
        output.write("round: %d\n" %(rd))
        output.write("Classification report for classifier %s:\n%s\n"
              % (model, metrics.classification_report(actual, predicted)))
        cm = metrics.confusion_matrix(actual, predicted)
        cm_avg += cm
        confusionMatrixFilename = 'confusion_matrix_' + str(rd)
        confusion_matrix_file = pd.DataFrame(cm)
        confusion_matrix_file.to_csv(confusionMatrixFilename, sep=',', index=False, header=True)
        plot_confusion_matrix(cm, label_names, rd)
        output.write("Confusion matrix:\n%s\n\n" % (cm))
        output.write("starting time: %s\n" %(str(t0)))
        output.write("ending time: %s\n" %(str(t1)))
        output.write("time consumed: %s\n\n\n" %(str(t1-t0)))
    except IOError:
        print('IOError')
###############################################################################

cm_avg /= rd
confusionMatrixFilename = 'confusion_matrix_avg'
confusion_matrix_file = pd.DataFrame(cm_avg)
confusion_matrix_file.to_csv(confusionMatrixFilename, sep=',', index=False, header=True)
plot_confusion_matrix(cm_avg, label_names, 'avg')
try:
    output.write("Confusion matrix:\n%s\n" % (cm_avg))
except IOError:
    print('IOError')

output.close()
print('end')

