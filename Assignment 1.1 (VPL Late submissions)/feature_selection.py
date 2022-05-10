import numpy as np
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
import time

def select_feature(train):
    train_data = np.loadtxt(train, dtype="str", delimiter=",")
    X_train = train_data[1:, 1:-1].astype(float)
    col = X_train.shape[1]
    row = X_train.shape[0]
    Y_train = train_data[1:,-1].astype(float)
    for i in range(col):
        feat = X_train[:,i]
        R = np.corrcoef(Y_train, feat)
        print(str(i) + ": " + str(R[1,0]))
def feature(train):
    # Training
    S = time.time()
    train_data = np.loadtxt(train, dtype=str, delimiter=',')
    X_train = train_data[1:, 1:-1].astype(float)
    row = X_train.shape[0]
    X_train = np.concatenate((np.ones((row,1)), X_train), axis = 1)
    row = X_train.shape[0]
    #feat_index = [0,3,5,6,7,8,9,10,11,12,13,16,17,20,21,23,24,25,26,28,29]
    feat_index = [11,14,18,22,24,16,21,23,4]
    feat_len = len(feat_index)
    for x in range(feat_len):
        i = feat_index[x]
        feat1 = 10000*X_train[:, i].reshape((row,1))
        for y in range(x+1, feat_len):
            if(x != y):
                j = feat_index[y]
                feat2 = X_train[:, j].reshape((row,1))
                feat = feat1 + feat2
                X_train = np.concatenate((X_train, feat), axis=1)


    X_train = np.delete(X_train, [4,14,16,18,20,22], 1)
    Y_train = train_data[1:,-1].astype(float)
    

    start = time.time()
    col = X_train.shape[1]
    D = {}
    List_of_dict = [D for i in range(col)]
    for i in range(1,col):
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

    
    end  = time.time()
    print(end-start)
    #pca = PCA(n_components= 23)
    #X_train = pca.fit_transform(X_train)
    start = time.time()
    print("time for pca")
    print(start - end)
    #poly = PolynomialFeatures(2)
    #X_train = poly.fit_transform(X_train)
    print("PCA POLY done")
    print("Features: " + str(X_train.shape))
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
    print(Del)
    
select_feature("../Assignment_1/data/train_large.csv")
