import time
import numpy as np
import pandas as pd
import sys
from scipy.special import softmax
import math
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectKBest, chi2

start = 0.0


def loss_funct(X_train, Y_train, Weight):
    Y_pred = prediction(X_train, Weight)
    eps=1e-15
    return abs(np.sum(np.log(np.clip(np.sum(Y_train*Y_pred, axis=1), eps, 1-eps)))/X_train.shape[0])
    #return abs(np.sum(np.log(np.sum(Y_train*Y_pred, axis=1)))/X_train.shape[0])

def Adaptive_alpha(X_train, Y_train, Weight, initial_n, alpha, beta):
    pred = prediction(X_train, Weight)
    grad_w = Calc_gradient(X_train, Y_train, pred)
    loss_w = loss_funct(X_train, Y_train, Weight)
    count = 0
    while(loss_funct(X_train, Y_train, Weight-initial_n*grad_w) > (loss_w - alpha*initial_n*(np.linalg.norm(grad_w)**2))):
        initial_n = initial_n*beta
    return initial_n


def prediction(X, Weight):
    temp = X@Weight
    out = softmax(temp.T, axis = 0)
    return out.T
    
def Calc_gradient(X_train, Y_train, P):
    row = X_train.shape[0]
    num = (X_train.T @ (P - Y_train))
    return num/row

def Weight_matrix(X_train, Y_train, X_test, Weight, strat, iterations, learning_rate, weight_file, output_file):
    myflag = True
    row_test = X_test.shape[0]
    for i in range(iterations):
        if(strat == 1):
            LR = float(learning_rate)
        if(strat == 2):
            LR = float(learning_rate)/ math.sqrt(i+1)
        if(strat == 3):
            par = learning_rate.split(',')
            initial_n = float(par[0])
            alpha = float(par[1])
            beta = float(par[2])
            LR = Adaptive_alpha(X_train, Y_train, Weight, initial_n, alpha, beta)
        pred = prediction(X_train, Weight)
        Grad = Calc_gradient(X_train, Y_train, pred)
        Weight = Weight - (Grad*LR)
        
        # if(strat == 3):
        #     if(myflag):
        #         myflag = False
        #         # row_test = X_test.shape[0]
        #         # X_test = np.concatenate((np.ones((row_test,1)), X_test), axis=1)
        #         pred = prediction(X_test, Weight)
        #         Y_pred = np.zeros((row_test, 1))
        #         for k in range(row_test):
        #             max_index = -1
        #             max_value = -1
        #             for l in range(8):
        #                 if(pred[k][l] > max_value):
        #                     max_value = pred[k][l]
        #                     max_index = l
        #             Y_pred[k] = max_index+1
        #         np.savetxt(weight_file, Weight, delimiter="\n")
        #         np.savetxt(output_file, Y_pred, delimiter="\n")
        #         temp_time = time.time()
        #         print(temp_time-start)
        #         print("iteration: " + str(i))
        #     else:
        #         curr_time = time.time()
        #         if(curr_time - temp_time > 60):
        #             # row_test = X_test.shape[0]
        #             # X_test = np.concatenate((np.ones((row_test,1)), X_test), axis=1)
        #             pred = prediction(X_test, Weight)
        #             Y_pred = np.zeros((row_test, 1))
        #             for k in range(row_test):
        #                 max_index = -1
        #                 max_value = -1
        #                 for l in range(8):
        #                     if(pred[k][l] > max_value):
        #                         max_value = pred[k][l]
        #                         max_index = l
        #                 Y_pred[k] = max_index+1
        #             curr_time = time.time()
        #             if(curr_time - start < 540):
        #                 np.savetxt(weight_file, Weight, delimiter="\n")
        #                 np.savetxt(output_file, Y_pred, delimiter="\n")
        #             temp_time = time.time()
        #             print(temp_time-start)
        #             print("iteration: " + str(i))

                
    return Weight

def Weight_matrix_b(X_train, Y_train, X_test, Weight, strat, iterations, learning_rate, size, weight_file, output_file, mode):
    X_batch = []
    Y_batch = []
    no_batch = X_train.shape[0]//size
    myflag = True
    row_test = X_test.shape[0]
    for i in range(no_batch):
        tempX = X_train[i*size:(i+1)*size, :]
        tempY = Y_train[i*size:(i+1)*size]
        X_batch.append(tempX)
        Y_batch.append(tempY)
    for i in range(iterations):
        if(strat == 1):
            LR = float(learning_rate)
        if(strat == 2):
            LR = float(learning_rate)/ math.sqrt(i+1)
        if(strat == 3):
            par = learning_rate.split(',')
            initial_n = float(par[0])
            alpha = float(par[1])
            beta = float(par[2])
            #print("Initial: " + par[0])
            LR = Adaptive_alpha(X_train, Y_train, Weight, initial_n, alpha, beta)
            #print(LR)
        for j in range(len(X_batch)):
            pred = prediction(X_batch[j], Weight)
            Grad = Calc_gradient(X_batch[j], Y_batch[j], pred)
            Weight = Weight - (Grad*LR)

            #temp_time = time.time()
            if(mode == "c"):
                if(myflag):
                    myflag = False
                    
                    # X_test = np.concatenate((np.ones((row_test,1)), X_test), axis=1)
                    pred = prediction(X_test, Weight)
                    Y_pred = np.zeros((row_test, 1))
                    for k in range(row_test):
                        max_index = -1
                        max_value = -1
                        for l in range(8):
                            if(pred[k][l] > max_value):
                                max_value = pred[k][l]
                                max_index = l
                        Y_pred[k] = max_index+1
                    np.savetxt(weight_file, Weight, delimiter="\n")
                    np.savetxt(output_file, Y_pred, delimiter="\n")
                    temp_time = time.time()
                    print(temp_time-start)
                    print("iteration: " + str(i))
                else:
                    curr_time = time.time()
                    if(curr_time - temp_time > 60):
            
                        # X_test = np.concatenate((np.ones((row_test,1)), X_test), axis=1)
                        pred = prediction(X_test, Weight)
                        Y_pred = np.zeros((row_test, 1))
                        for k in range(row_test):
                            max_index = -1
                            max_value = -1
                            for l in range(8):
                                if(pred[k][l] > max_value):
                                    max_value = pred[k][l]
                                    max_index = l
                            Y_pred[k] = max_index+1
                        curr_time = time.time()
                        if(curr_time - start < 540):
                            np.savetxt(weight_file, Weight, delimiter="\n")
                            np.savetxt(output_file, Y_pred, delimiter="\n")
                        temp_time = time.time()
                        print(temp_time-start)
                        print("iteration: " + str(i))
    return Weight


def cross_validation(X_train, Y_train, X_test, param, weight_file, output_file, SkBest, Y_true):
    row = X_train.shape[0]
    R = row//10
    accu = 0.0
    for j in range(10):
        KX_test = X_train[j*R:(j+1)*R,:]
        KY_test = Y_train[j*R:(j+1)*R]
        KX_train = np.concatenate((X_train[0:j*R, :], X_train[(j+1)*R: , :]))
        KY_train = np.concatenate((Y_train[0:j*R], Y_train[(j+1)*R:]))
        KY_true =  Y_true[j*R:(j+1)*R]
        obj = SelectKBest(chi2, k=SkBest)
        KX_train_f = obj.fit_transform(KX_train, KY_train)
        KX_test_f = obj.transform(KX_test)
        strat = 2
        learning_rate = 10
        iterations = 250
        batches = 100
        Weight = np.zeros((KX_train_f.shape[1], KY_train.shape[1]))
        Weight = Weight_matrix_b(KX_train_f, KY_train, KX_test_f,  Weight, strat, iterations, learning_rate, batches, weight_file, output_file, "d")
    
        KY_pred = prediction(KX_test_f, Weight)
        # true_Y = np.zeros((KY_test.shape[0], 1))
        pred_Y = np.zeros((KY_pred.shape[0], 1))
        row_true = pred_Y.shape[0]
        col_true = pred_Y.shape[1]
        # for k in range(row_true):
        #     for l in range(col_true):
        #         if(KY_test[k][l] == 1):
        #             true_Y[k] = l+1
        #             break
        for k in range(row_true):
            index = np.argmax(KY_pred[k])
            # max_pred = -1
            # index = -1
            # for l in range(col_true):
            #     if(KY_pred[k][l] > max_pred):
            #         max_pred = KY_pred[k][l]
            #         index = l
            pred_Y[k] = index + 1
        accu += f1_score(KY_true, pred_Y, average='micro')
    print("Accuracy: " + str(accu/10))





def logistic(X_train, Y_train, X_test, param, weight_file, output_file, mode, Y_true):
    row = X_train.shape[0]
    X_train = np.concatenate((np.ones((row,1)), X_train), axis=1)
    row_test = X_test.shape[0]
    X_test = np.concatenate((np.ones((row_test,1)), X_test), axis=1)
    
    if(mode == "a"):
        f  = open(param, 'r')
        parameters = f.readlines()
        f.close()
        strat = int(parameters[0])
        learning_rate = parameters[1]
        iterations = int(parameters[2])
    
        Weight = np.zeros((X_train.shape[1], Y_train.shape[1]))
        Weight = Weight_matrix(X_train, Y_train, X_test, Weight, strat, iterations, learning_rate, weight_file, output_file)
        
        np.savetxt(weight_file, Weight, delimiter="\n")

    elif(mode == "b"):
        f  = open(param, 'r')
        parameters = f.readlines()
        f.close()
        strat = int(parameters[0])
        learning_rate = parameters[1]
        iterations = int(parameters[2])
        batches = int(parameters[3])
        Weight = np.zeros((X_train.shape[1], Y_train.shape[1]))
        Weight = Weight_matrix_b(X_train, Y_train, X_test, Weight, strat, iterations, learning_rate, batches, weight_file, output_file, mode)
        
        np.savetxt(weight_file, Weight, delimiter="\n")

    elif(mode == "c"):
         # Obtained from param_search.py
        strat = 2
        #learning_rate = "2.5,0.2,0.5"
        learning_rate = '10'
        iterations = 250
        batches = 100  
        Weight = np.zeros((X_train.shape[1], Y_train.shape[1]))
        Weight = Weight_matrix_b(X_train, Y_train, X_test,  Weight, strat, iterations, learning_rate, batches, weight_file, output_file, mode)
        
    elif(mode == "d"):
        SkBest = [50, 100, 200, 400, 600, 800]
        for i in range(len(SkBest)):
            cross_validation(X_train, Y_train, X_test, param, weight_file, output_file, SkBest[i], Y_true)

        # strat = 2
        # learning_rate = 10
        # iterations = 250
        # batches = 100
        #feature selection  
        

        #Weight = np.zeros((X_train.shape[1], Y_train.shape[1]))
        #Weight = Weight_matrix_b(X_train, Y_train, X_test,  Weight, strat, iterations, learning_rate, batches, weight_file, output_file)
    
    
    if(mode == "a" or mode == "b"):

        pred = prediction(X_test, Weight)
        Y_pred = np.zeros((row_test, 1))
        for i in range(row_test):
            max_index = -1
            max_value = -1
            for j in range(8):
                if(pred[i][j] > max_value):
                    max_value = pred[i][j]
                    max_index = j
            Y_pred[i] = max_index+1
        np.savetxt(output_file, Y_pred, delimiter="\n")

    



if __name__ == '__main__':
    start = time.time()
    mode = "d"
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    param = ""
    output_file = ""
    weight_file = ""
    
    train = pd.read_csv(train_file, index_col = 0)    
    test = pd.read_csv(test_file, index_col = 0)
    Y_train = np.array(train['Length of Stay'])

    train = train.drop(columns = ['Length of Stay'])
    #Ensuring consistency of One-Hot Encoding

    data = pd.concat([train, test], ignore_index = True)
    cols = train.columns
    cols = cols[:-1]

    data = pd.get_dummies(data, columns=cols, drop_first=True)
    data = data.to_numpy()
    X_train = data[:train.shape[0], :].astype(float)
    X_test = data[train.shape[0]:, :].astype(float)
    
    Y_true = Y_train
    Y_train = pd.get_dummies(Y_train)

    Y_train = Y_train.to_numpy()
    Y_train = Y_train.astype(float)

    logistic(X_train, Y_train, X_test, param, weight_file, output_file, mode, Y_true)

    end = time.time()
    print(end-start)


