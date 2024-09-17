import os
import pickle
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch

## SAVE AND LOAD ##
def ensureDirectoryExists(dir_name):
    assert dir_name[-1] == '/', 'directory name must end with /'
    if os.path.exists(dir_name):
        exists=True
        if os.path.isdir(dir_name)==False:
            assert False,'Non-directory file '+dir_name+' exists.'
    else: 
        exists=False
        os.makedirs(dir_name)
    return exists

def pickleLoad(path):
    if os.path.exists(path): 
        with open(path,'rb') as f: data=pickle.load(f)
        print('loading data from',path)
        return data
    else: 
        print('Load failed. File does not exist:', path)
        return False
    
def pickleSave(data,dir_name,fileName,overwrite=True):
    path=dir_name+fileName
    if os.path.exists(path): 
        if overwrite: saveData=True ## FILE EXISTS, OVERWRITE
        else: saveData=False ##FILE EXISTS, DO NOT OVERWRITE
    else: saveData=True ##FILE DOES NOT EXIST
    if saveData: 
        ensureDirectoryExists(dir_name)
        with open(path,'wb') as f: pickle.dump(data,f)
        print('saved to',path)
####################


## SPLIT DATA ##
def splitDataAndLoadToDevice(X,y,seed,device):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    # Convert class labels to integers
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    # Convert the data to PyTorch tensors
    X_train_ = torch.tensor(X_train, dtype=torch.float).to(device)
    y_train_ = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_ = torch.tensor(X_test, dtype=torch.float).to(device)
    y_test_ = torch.tensor(y_test, dtype=torch.long).to(device)
    
    return X_train_, X_test_, y_train_, y_test_

def splitKfAndLoadToDevice(X,y,n_splits,seed,device, to_tensor=True):
    kf = KFold(n_splits=n_splits,shuffle=True,random_state = seed)  
    X_trains = []
    X_tests = []
    y_trains = []
    y_tests = []
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if to_tensor:
            X_train = torch.tensor(X_train, dtype=torch.float).to(device)
            y_train = torch.tensor(y_train, dtype=torch.long).to(device)
            X_test = torch.tensor(X_test, dtype=torch.float).to(device)
            y_test = torch.tensor(y_test, dtype=torch.long).to(device)
        X_trains.append(X_train)
        X_tests.append(X_test)
        y_trains.append(y_train)
        y_tests.append(y_test)
    return X_trains, X_tests, y_trains, y_tests