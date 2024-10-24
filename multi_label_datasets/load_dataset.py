
from sklearn.datasets import load_wine
import numpy as np
import pandas as pd
from skmultilearn.dataset import load_dataset
from sklearn.model_selection import train_test_split
import scipy.io

import os

datasets = [
    'medical',
]


class DatasetModel:
    def __init__(self, dataset_name, x_train, y_train, x_test, y_test,
                 number_of_features, number_of_labels, all_x, all_y):
        self.dataset_name = dataset_name
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.number_of_features = number_of_features
        self.number_of_labels = number_of_labels
        self.all_x = all_x 
        self.all_y = all_y

def load_my_dataset(dataset_name=""):
        
        try:

            x_train = np.loadtxt('./multi_label_datasets/saved/'+dataset_name+'_x_train.txt')
            y_train =np.loadtxt('./multi_label_datasets/saved/'+dataset_name+'_y_train.txt')
            # feature_names =np.loadtxt('./multi_label_datasets/saved/'+dataset_name+'_feature_names.txt')
            # label_names = np.loadtxt('./multi_label_datasets/saved/'+dataset_name+'_label_names.txt')
            x_test = np.loadtxt('./multi_label_datasets/saved/'+dataset_name+'_x_test.txt')
            y_test = np.loadtxt('./multi_label_datasets/saved/'+dataset_name+'_y_test.txt')
            
        except:
            
            x_train, y_train, feature_names, label_names = load_dataset(dataset_name, 'train')
            x_test, y_test, _, _ = load_dataset(dataset_name, 'test')
            x_train = x_train.toarray()
            y_train = y_train.toarray()
            x_test = x_test.toarray()
            y_test = y_test.toarray()
            np.savetxt('./multi_label_datasets/saved/'+dataset_name+'_x_train.txt', x_train)
            np.savetxt('./multi_label_datasets/saved/'+dataset_name+'_y_train.txt', y_train)
            # np.savetxt('./multi_label_datasets/saved/'+dataset_name+'_feature_names.txt', feature_names)
            # np.savetxt('./multi_label_datasets/saved/'+dataset_name+'_label_names.txt', label_names)
            np.savetxt('./multi_label_datasets/saved/'+dataset_name+'_x_test.txt', x_test)
            np.savetxt('./multi_label_datasets/saved/'+dataset_name+'_y_test.txt', y_test)

        
        all_x = np.concatenate((x_train,x_test))
        all_y = np.concatenate((y_train,y_test))

        number_of_features = all_x.shape[1]
        number_of_labels = all_y.shape[1]

        datasetModel = DatasetModel(dataset_name, x_train, y_train, x_test, y_test, number_of_features, number_of_labels,all_x,all_y)
                            
            
        return datasetModel


