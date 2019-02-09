import os
import glob
import numpy as np
import pandas as pd
import datetime
import operator
import numpy.ma as ma
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn import decomposition
from xgboost import XGBClassifier




def load_train_file(path):
    # Loads training data
    '''

    :param path: Location of the training file
    :return: Features: X , Labels : Y
    '''
    train_dataset = pd.read_csv (path, sep=" ", header=None, low_memory=False)
    train_dataset.replace ('?', np.nan, inplace=True)
    train_dataset = train_dataset.fillna(0)
    train_values = train_dataset.values.astype(np.float64)
    X = train_values[:, 0:205]
    Y = train_values[:, 205]
    return X, Y


def loadX_test_file(path):
    #Loads features of test data
    '''

    :param path:  Location of the testing file
    :return: Features of test data
    '''
    X_dataset = pd.read_csv (path, sep=" ", header=None, low_memory=False)
    X_dataset.replace ('?', np.nan, inplace=True)
    X_dataset = X_dataset.fillna(0)
    X_values = X_dataset.values.astype(np.float64)
    X_values = X_values[:, 0:205]
    return X_values


def loadY_test_file(path):
    #Loads class of preliminary data
    '''

    :param path:  Location of preliminary data
    :return: Class of preliminary data
    '''
    Y_dataset = pd.read_csv(path, sep="\n", header=None, low_memory=False)
    Y_dataset = np.transpose(Y_dataset.values)[0]
    Y_dataset = Y_dataset.astype(np.float64)
    return Y_dataset


def load_train():
    #Loads training data

    """

    :return: Training features
             Training class
    """
    folders = ['train_mv', 'train_nmv', 'prelim']
    X_train = []
    y_train = []
    train_files = []
    for folder in folders:
        index = folders.index (folder)
        print("Loading folder {} (Index : {})".format (folder, index))
        path = os.path.join (os.path.dirname (__file__), 'train', folder, '*.txt')
        files = glob.glob (path)
        if (folder == "prelim"):
            for file in files:
                file_base = os.path.basename(file)
                if(file_base == 'prelim-gold.txt'):
                     Y_dataset = loadY_test_file(file)
                     for dataset in Y_dataset:
                         y_train.append(dataset)
                elif file_base=='prelim-mv-noclass':
                     X_dataset = loadX_test_file(file)
                     train_files.append(file_base)
                     for dataset in X_dataset:
                        X_train.append(dataset)
        else:
            for file in files:
                file_base = os.path.basename(file)
                X_dataset, Y_dataset = load_train_file(file)
                for index, dataset in enumerate (X_dataset):
                    X_train.append(dataset)
                    y_train.append(Y_dataset[index])

                train_files.append (file_base)

    return X_train, y_train, train_files







def load_test():
    #Loads testing data
    '''
    :returns: X_test features for test data
    '''
    path = os.path.join (os.path.dirname (__file__), 'test', '*.txt')
    files = glob.glob(path)
    X_test = []
    test_filename = []
    for file in files:
        file_base = os.path.basename(file)
        X_test.append(loadX_test_file(file))
        test_filename.append(file_base)
    return X_test, test_filename


def createModel(train_data, nfolds, random_state = 42):
    #Creates model

    '''
    :param train_data: Training data
    :param nfolds: num_folds
    :param random_state:
    :return: model
    '''
    model = KFold (len(train_data), n_folds=nfolds, shuffle=True, random_state=random_state)
    return model

def create_submission(predictions,test_name):
    #Creates a predictions output file
    '''
    :param predictions: list of predictions
    '''
    for idx, prediction in enumerate(predictions):
        with open(test_name[idx]+'predictions.txt', 'w') as f:
             for item in predictions[idx]:
                    f.write ("%s\n" % int(item))


def read_and_normalize_train_data():
    #standardizes training data into same scale
    '''
    :return: Scaled training data, training labels, names of training files
    '''
    X, Y, train_files = load_train ()
    scaler = StandardScaler().fit (X)
    X = scaler.transform(X)
    return X, Y, scaler, train_files

def read_and_normalize_test_data(scaler):
    # standardizes testing data into same scale
    '''
    :return: Scaled testing data, names of testing files
    '''
    test_X_array,test_files = load_test()
    for idx, test_X in enumerate(test_X_array):
        test_X_array[idx] = scaler.transform(test_X)

    return test_X_array, test_files


def run_cross_validation(nfolds=10):
    #Implements the main algorithm
    '''
    :param nfolds: number of folds
    :return: index of classifier that gives the best accuracy in training instances, list of classifiers
             ,scaler used to scale training data
    '''


    train_data, train_labels, scaler, train_files = read_and_normalize_train_data ()
    classifiers = []
    accuracy_percent = []
    #Hyperparameters
    random_state = 51
    n_estimators = 100
    max_depth = 6
    sub_sample = 0.8

    for x in range(10):
        #10 iterations of 7 folds
        num_fold = 0
        kf = createModel(train_data,nfolds,random_state)
        random_state = random_state + 1
        for train_index, test_index in kf:
            classifier = XGBClassifier (learning_rate=0.1,
                                        n_estimators=100,
                                        max_depth=8,
                                        min_child_weight=1,
                                        gamma=0,
                                        subsample=0.8,
                                        colsample_bytree=0.8,
                                        objective='binary:logistic',
                                        nthread=4,
                                        scale_pos_weight=1,

                                       seed=27)
            # randomly changing values of parameters to get best model
            n_estimators += 200
            if n_estimators > 1000:
                n_estimators = 100
            max_depth += 1
            if max_depth == 10:
                max_depth = 6
            sub_sample -= 0.01
            if sub_sample < 0.7:
                sub_sample = 0.8

            x_train = train_data[train_index]
            y_train = np.array (train_labels)[train_index]
            x_valid = train_data[test_index]
            y_valid = np.array (train_labels)[test_index]
            classifier.fit (x_train, y_train)

            y_pred = classifier.predict(x_valid)
            accuracy = round(100*accuracy_score (y_valid, y_pred),2)
            print ("Num fold: {} , Iteration : {}, Accuracy of XGboost: {} ".format(num_fold ,x, accuracy))
            classifiers.append(classifier)
            accuracy_percent.append(accuracy)
            num_fold = num_fold + 1

    max_accuracy_index, max_accuracy = max (enumerate (accuracy_percent), key=operator.itemgetter (1))
    print("Train : max accuracy:{} and index: {} ".format(max_accuracy,max_accuracy_index))
    return max_accuracy_index,classifiers, scaler


def run_cross_validation_test(index, scaler, classifiers):
    #Test and classifies unknown data
    #Calls create_submission function to create file for predicted output
    '''
    :param index: Index of best classifier
    :param scaler: Scaler used in scaling training data
    :param classifiers: List of all classifiers
    '''

    test_X, test_name = read_and_normalize_test_data (scaler)
    predictions = []
    classifier = classifiers[index]
    for test_X_data in test_X:
         y_pred = classifier.predict(test_X_data)
         predictions.append(y_pred)

    create_submission(predictions,test_name)



if __name__ == '__main__':
    num_folds = 7
    index_classifier, classifiers, scaler = run_cross_validation(num_folds)
    run_cross_validation_test( index_classifier, scaler, classifiers)




