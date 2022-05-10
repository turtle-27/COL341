import numpy as np
import sys
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
import time

lamda = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 10, 30]
def feature(train):
    # Training
    S = time.time()
    train_data = np.loadtxt(train, dtype=str, delimiter=',')
    X_train = train_data[1:, 1:-1].astype(float)
    row = X_train.shape[0]
    feat_index = [1,3,10,12]
    for i in feat_index:
        for j in feat_index:
            if(i != j):
                feat1 = 10000*X_train[:, i].reshape(row,1)
                feat2 = X_train[:, j].reshape(row,1)
                feat = feat1 + feat2
                X_train = np.concatenate((X_train, feat), axis=1)


    X_train = np.delete(X_train, [2,4,14,16,18,20,22], 1)
    Y_train = train_data[1:,-1].astype(float)
    

    # feat = X_train[:, 11].reshape(row,1)
    # feat = np.square(feat)

    start = time.time()
    col = X_train.shape[1]
    D = {}
    List_of_dict = [D for i in range(col)]
    for i in range(col):
        Dict = {}
        Count = {}
        for j in range(row):
            key = X_train[j,i]
            if(key in Dict):
                Dict[key] += Y_train[j]
                Count[key] += 1
            else:
                Dict[key] = Y_train[j]
                Count[key] = 1
        for key in Dict:
            Dict[key] = Dict[key]/Count[key]
        List_of_dict[i] = Dict
        for j in range(row):
            X_train[j, i] = Dict[X_train[j,i]]

    print("Target encoding done")

    X_train = np.concatenate((np.ones((row,1)), X_train), axis = 1)
    end  = time.time()
    print(end-start)
    pca = PCA(n_components= 17)
    X_train = pca.fit_transform(X_train)
    start = time.time()
    print("time for pca")
    print(start - end)
    poly = PolynomialFeatures(2)
    X_train = poly.fit_transform(X_train)
    print("PCA POLY done")
    end = time.time()
    print(end - start)
    lamda = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 10, 30]
    min_lamda = -1
    max_score = -1
    R  = row//10
    for i in range(len(lamda)):
        score = 0.0
        for j in range(10):
            KX_test = X_train[j*R:(j+1)*R,:]
            KY_test = Y_train[j*R:(j+1)*R]
            KX_train = np.concatenate((X_train[0:j*R, :], X_train[(j+1)*R: , :]))
            KY_train = np.concatenate((Y_train[0:j*R], Y_train[(j+1)*R:]))
            model = linear_model.LassoLars(lamda[i])
            model.fit(KX_train, KY_train)
            score += model.score(KX_test, KY_test)
        score = score/10
        if(max_score == -1):
            max_score = score
            min_lamda = lamda[i]
        elif(max_score < score): 
            max_score = score
            min_lamda = lamda[i]
        print("score:" + str(score))
    print("LAMBDA: ", str(min_lamda))
    model = linear_model.LassoLars(min_lamda)
    model.fit(X_train, Y_train)
    W = model.coef_
    Del = []
    f = open("Del.txt", "w")
    for i in range(len(W)):
        if(W[i] == 0):
            Del.append(i)
            f.write(str(i)+"\n")
    f.close()
    
feature("../Assignment_1/data/train.csv")