import numpy as np
import sys


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
    train_data = np.loadtxt(train, dtype=str, delimiter=',')
    X_train = train_data[1:, 1:-1].astype(float)
    row = X_train.shape[0]
    X_train = np.concatenate((np.ones((row,1)), X_train), axis = 1)
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

   
    # Testing

    X_train_T = X_train.T
    W = (np.linalg.pinv(X_train_T@X_train))@(X_train_T@Y_train)
    test_data = np.loadtxt(test, dtype=str, delimiter=',')
    X_test = test_data[1:, 1:].astype(float)
    row  = X_test.shape[0]
    X_test = np.concatenate((np.ones((row,1)), X_test), axis = 1)
    row  = X_test.shape[0]
    col = X_test.shape[1]

    for x in range(feat_len):
        i = feat_index[x]
        feat1 = 10000*X_test[:, i].reshape((row,1))
        for y in range(x+1, feat_len):
            if(x != y):
                j = feat_index[y]
                feat2 = X_test[:, j].reshape((row,1))
                feat = feat1 + feat2
                X_test = np.concatenate((X_test, feat), axis=1)

    X_test = np.delete(X_test, [4,14,16,18,20,22], 1)
    row = X_test.shape[0]
    col = X_test.shape[1]
    for i in range(1,col):
        Dict = List_of_dict[i]
        for j in range(row):
            key = X_test[j, i]
            if(key in Dict):
                X_test[j, i] = Dict[key]
            else:
                X_test[j,i] = 0

    Y_pred = X_test@W
    np.savetxt(output, Y_pred, delimiter="\n")


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
