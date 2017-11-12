'''
Created on Jun 3, 2017

@author: yang
'''
import numpy as np
from sklearn.decomposition import PCA
from time import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from scipy import io
import scipy
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

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
    plt.title('Confusion Matrix '+str(i))
    #plt.show()
    plt.savefig('confusion_matrix_'+str(i)+'.png', format='png')


# Get data metrix.
raw_data_1_dict_path = "cifar-10-batches-py/data_batch_1"
raw_data_1_dict = unpickle(raw_data_1_dict_path)
raw_data_1 = raw_data_1_dict.get(b'data')

raw_data_2_dict_path = "cifar-10-batches-py/data_batch_2"
raw_data_2_dict = unpickle(raw_data_2_dict_path)
raw_data_2 = raw_data_2_dict.get(b'data')

raw_data_3_dict_path = "cifar-10-batches-py/data_batch_3"
raw_data_3_dict = unpickle(raw_data_3_dict_path)
raw_data_3 = raw_data_3_dict.get(b'data')

raw_data_4_dict_path = "cifar-10-batches-py/data_batch_4"
raw_data_4_dict = unpickle(raw_data_4_dict_path)
raw_data_4 = raw_data_4_dict.get(b'data')

raw_data_5_dict_path = "cifar-10-batches-py/data_batch_5"
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
n_classes = label_names.shape[0]

# set output file path
result_file = 'knn_report.csv'
output = open(result_file, 'a')

# set the proportion of training data and validation data
kf = KFold(n_splits=10)

# set a variable to record the number of cross-validation performed
rd = 0

print("start")

# set a variable to record average confusion matrix
cm_avg = np.zeros((n_classes,n_classes))

# train classifier for 10-fold cross-validation
for train, valid in kf.split(X):
    t0 = time()
    
    rd += 1
    print(rd)
    
# split data into training data and validation data    
    X_train, X_valid, y_train, y_valid = X[train], X[valid], y[train], y[valid]
    
# Perform PCA on data
    n_components = 100
    pca = PCA(n_components=n_components, svd_solver='randomized',
              whiten=True).fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_valid_pca = pca.transform(X_valid)
            
###############################################################################
# The KNN classifier.
    n_neighbors = 80
    classifier = KNeighborsClassifier(n_neighbors)
    classifier.fit(X_train_pca, y_train)
    
# predict validation data             
    actual = y_valid
    predicted = classifier.predict(X_valid_pca)
    
    t1 = time()
    
# output results       
    try:
        output.write("round: %d\n\n" %(rd))
        output.write("Classification report for classifier %s:\n%s\n"
              % (classifier, metrics.classification_report(actual, predicted)))
        cm = metrics.confusion_matrix(actual, predicted)
        cm_avg += cm
        confusionMatrixFilename = 'confusion_matrix_' + str(rd)
        confusion_matrix_file = pd.DataFrame(cm)
        confusion_matrix_file.to_csv(confusionMatrixFilename, sep=',', index=False, header=True)
        plot_confusion_matrix(cm, label_names, rd)
        output.write("Confusion matrix:\n%s\n" % (cm))
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
