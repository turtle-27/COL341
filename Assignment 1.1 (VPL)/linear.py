import numpy as np
import sys
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
import time

def partA(train, test, outputfile, weightfile):
    # Training
    
    train_data = np.loadtxt(train,dtype=str,delimiter=',')
    X_train = train_data[1:, 1:-1].astype(float)
    row = X_train.shape[0]
    X_train = np.concatenate((np.ones((row,1)), X_train), axis = 1)
    Y_train = train_data[1:,-1].astype(float)
    X_T = X_train.T
    W = (np.linalg.inv(X_T@X_train))@(X_T@Y_train)
    np.savetxt(weightfile, W, delimiter = "\n")
    # Testing
    
    test_data = np.loadtxt(test, dtype=str, delimiter=',')
    X_test = test_data[1:, 1:].astype(float)
    row = X_test.shape[0]
    X_test = np.concatenate((np.ones((row,1)), X_test), axis = 1)
    Y_pred = X_test@W
    np.savetxt(outputfile, Y_pred, delimiter="\n")

#partA("../Assignment_1/data/train.csv", "../Assignment_1/data/test.csv")

def get_error(A, B):
    return np.sum((np.square(np.subtract(A,B))))/np.sum((np.square(B)))
def partB(train, test, L, outputfile, weightfile, bestparameter):
    # Training
    train_data = np.loadtxt(train, dtype=str, delimiter=',')
    X_train = train_data[1:, 1:-1].astype(float)
    row = X_train.shape[0]
    X_train = np.concatenate((np.ones((row,1)), X_train), axis = 1)
    X_train_T = X_train.T
    feature = X_train.shape[1]
    Y_train = train_data[1:,-1].astype(float)
    lamda =  np.loadtxt(L,delimiter='\n')
    min_lamda = -1
    min_error = -1
    R  = row//10
    for i in range(len(lamda)):
        error = 0.0
        for j in range(10):
            KX_test = X_train[j*R:(j+1)*R,:]
            KY_test = Y_train[j*R:(j+1)*R]
            KX_train = np.concatenate((X_train[0:j*R, :], X_train[(j+1)*R: , :]))
            KY_train = np.concatenate((Y_train[0:j*R], Y_train[(j+1)*R:]))
            KX_train_T = KX_train.T
            I = lamda[i]*np.identity(feature)
            W = (np.linalg.inv(I + KX_train_T@KX_train))@(KX_train_T@KY_train)
            KY_pred = KX_test@W
            error += get_error(KY_pred,KY_test)
        if(min_error == -1):
            min_error = error
            min_lamda = lamda[i]
        elif(min_error > error):
            min_error = error
            min_lamda = lamda[i]
    # Testing
    test_data = np.loadtxt(test, dtype=str, delimiter=',')
    X_test = test_data[1:, 1:].astype(float)
    row = X_test.shape[0]
    X_test = np.concatenate((np.ones((row,1)), X_test), axis = 1)
    I = min_lamda*np.identity(feature)
    W = (np.linalg.inv(I + X_train_T@X_train))@(X_train_T@Y_train)
    np.savetxt(weightfile, W, delimiter = "\n")
    Y_pred = X_test@W
    np.savetxt(outputfile, Y_pred, delimiter="\n")
    f = open(bestparameter, "w")
    f.write(str(min_lamda))


def partC(train, test, output):
    # Training
    S = time.time()
    train_data = np.loadtxt(train, dtype=str, delimiter=',')
    X_train = train_data[1:, 1:-1].astype(float)
    row = X_train.shape[0]
    # feat_index = [1,3,10,12]
    # for x in range(4):
    #     for y in range(i+1, 4):
    #         if(x != y):
    #             i = feat_index[x]
    #             j = feat_index[y]
    #             feat1 = 10000*X_train[:, i].reshape((row,1))
    #             feat2 = X_train[:, j].reshape((row,1))
    #             feat = feat1 + feat2
    #             X_train = np.concatenate((X_train, feat), axis=1)


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
    # lamda = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 10, 30]
    # min_lamda = -1
    # max_score = -1
    # R  = row//10
    # for i in range(len(lamda)):
    #     score = 0.0
    #     for j in range(10):
    #         KX_test = X_train[j*R:(j+1)*R,:]
    #         KY_test = Y_train[j*R:(j+1)*R]
    #         KX_train = np.concatenate((X_train[0:j*R, :], X_train[(j+1)*R: , :]))
    #         KY_train = np.concatenate((Y_train[0:j*R], Y_train[(j+1)*R:]))
    #         model = linear_model.LassoLars(lamda[i])
    #         model.fit(KX_train, KY_train)
    #         score += model.score(KX_test, KY_test)
    #     score = score/10
    #     if(max_score == -1):
    #         max_score = score
    #         min_lamda = lamda[i]
    #     elif(max_score < score): 
    #         max_score = score
    #         min_lamda = lamda[i]
    #     print("score:" + str(score))
    # model = linear_model.LassoLars(min_lamda)
    # model.fit(X_train, Y_train)
    # W = model.coef_
    # Del = []
    # for i in range(len(W)):
    #     if(W[i] == 0):
    #         Del.append(i)
    # Testing
    Del = [0, 9, 12, 26, 58, 59, 63, 68, 73, 74, 75, 77, 85, 89, 93, 97, 114, 117, 126, 129, 133, 138, 139, 142, 161]
    X_train = np.delete(X_train, Del, 1)
    X_train_T = X_train.T
    W = (np.linalg.inv(X_train_T@X_train))@(X_train_T@Y_train)

    test_data = np.loadtxt(test, dtype=str, delimiter=',')
    X_test = test_data[1:, 1:-1].astype(float)
    row  = X_test.shape[0]
    col = X_test.shape[1]
    # feat_index = [1,3,10,12]
    # for x in range(4):
    #     for y in range(i+1, 4):
    #         if(x != y):
    #             i = feat_index[x]
    #             j = feat_index[y]
    #             feat1 = X_test[:,i].reshape((row,1))
    #             feat1 = 10000*feat1
    #             feat2 = X_test[:, j].reshape((row,1))
    #             feat = feat1 + feat2
    #             X_test = np.concatenate((X_test, feat), axis=1)

    X_test = np.delete(X_test, [2,4,14,16,18,20,22], 1)
    row = X_test.shape[0]
    col = X_test.shape[1]
    for i in range(col):
        Dict = List_of_dict[i]
        for j in range(row):
            key = X_test[j, i]
            if(key in Dict):
                X_test[j, i] = Dict[key]
            else:
                X_test[j,i] 

    X_test = np.concatenate((np.ones((row,1)), X_test), axis = 1)
    X_test = pca.transform(X_test)
    X_test = poly.transform(X_test)
    X_test = np.delete(X_test, Del, 1)
    Y_pred = X_test@W
    np.savetxt(output, Y_pred, delimiter="\n")
    E = time.time()
    print(E-S)
#print(partB("../Assignment_1/data/train.csv", "../Assignment_1/data/test.csv", "lambda.txt"))
#partC("../Assignment_1/data/train.csv", "lamdac.txt")               

# F = "../Assignment_1/data/train.csv"
# t = np.loadtxt(F, dtype=str, delimiter=',')
# Y_actual = t[1:,-1].astype(float)
# np.savetxt("gold_c.txt", Y_actual, delimiter="\n")



if __name__ == '__main__':
    mode = sys.argv[1]
    if(mode == "a"):
        train_file = sys.argv[2]
        test_file = sys.argv[3]
        output_file = sys.argv[4]
        weight_file = sys.argv[5]
        partA(train_file, test_file, output_file, weight_file)
    elif(mode == "b"):
        train_file = sys.argv[2]
        test_file = sys.argv[3]
        regularisation = sys.argv[4]
        output_file = sys.argv[5]
        weight_file = sys.argv[6]
        bestparameter = sys.argv[7]
        partB(train_file, test_file, regularisation, output_file, weight_file, bestparameter)
    elif(mode == "c"):
        train_file = sys.argv[2]
        test_file = sys.argv[3]
        output_file = sys.argv[4]
        partC(train_file, test_file, output_file)
    else:
        print("Please select a valid mode")
