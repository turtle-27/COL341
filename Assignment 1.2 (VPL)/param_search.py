from matplotlib.legend import Legend
import numpy as np
import pandas as pd
import sys
from scipy.special import softmax
import math
import time
from matplotlib import pyplot as plt 
from matplotlib import pyplot as plt2

flag = True
def loss_funct(X_train, Y_train, Weight):
    Y_pred = prediction(X_train, Weight)
    eps=1e-15
    re = abs(np.sum(np.log(np.clip(np.sum(Y_train*Y_pred, axis=1), eps, 1-eps)))/X_train.shape[0])
    global flag
    if(flag):
        print(re)
        flag = False
    return re
    #return abs(np.sum(np.log(np.clip(np.sum(Y_train*Y_pred, axis=1), eps, 1-eps)))/X_train.shape[0])

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


def Weight_matrix(X_train, Y_train, Weight, strat, iterations, learning_rate, output):
    loss = []
    total_time = []
    start = time.time()
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
        if(i%10 == 0):
            print(Weight)
            end = time.time()
            time_taken = end - start
            temp_loss = loss_funct(X_train, Y_train, Weight)
            total_time.append(time_taken)
            loss.append(temp_loss)
        pred = prediction(X_train, Weight)
        Grad = Calc_gradient(X_train, Y_train, pred)
        Weight = Weight - (Grad*LR)
    df = pd.DataFrame(data = {'Loss': np.array(loss), 'Runtime' : np.array(total_time)})
    df.to_csv(output)
    return Weight

def Weight_matrix_b(X_train, Y_train, Weight, strat, iterations, learning_rate, size, output):
    X_batch = []
    Y_batch = []
    no_batch = X_train.shape[0]//size
    for i in range(no_batch):
        tempX = X_train[i*size:(i+1)*size, :]
        tempY = Y_train[i*size:(i+1)*size]
        X_batch.append(tempX)
        Y_batch.append(tempY)

    loss = []
    total_time = []
    start = time.time()
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
        if(i%10 == 0):
            end = time.time()
            time_taken = end - start
            temp_loss = loss_funct(X_train, Y_train, Weight)
            total_time.append(time_taken)
            loss.append(temp_loss)
        for j in range(len(X_batch)):
            pred = prediction(X_batch[j], Weight)
            Grad = Calc_gradient(X_batch[j], Y_batch[j], pred)
            Weight = Weight - (Grad*LR)
    df = pd.DataFrame(data = {'Loss': np.array(loss), 'Runtime' : np.array(total_time)})
    df.to_csv(output)   
    return Weight


def logistic_beta(X_train, Y_train, param, mode, output):
    row = X_train.shape[0]
    X_train = np.concatenate((np.ones((row,1)), X_train), axis=1)

    if(mode == "b"):
        #strat = int(parameters[0])
        strat = 3
        #learning_rate = parameters[1]
        learning_rate  = ['2.5,0.5,0.1', '2.5, 0.5, 0.3', '2.5, 0.5, 0.5', '2.5, 0.5, 0.7', '2.5, 0.5, 0.9']
        rate  = ['0.1', '0.3', '0.5', '0.7', '0.9']
        #iterations = int(parameters[2])
        iterations = 250
        batches = 200
        #batches = [100,200,400,800,1600]
       
        for i in range(len(learning_rate)):
            global flag 
            flag = True
            Weight = np.zeros((X_train.shape[1], Y_train.shape[1]))
            name = output + "_" + rate[i] + ".csv"
            Weight = Weight_matrix_b(X_train, Y_train, Weight, strat, iterations, learning_rate[i], batches, name)
        

def logistic_alpha(X_train, Y_train, param, mode, output):
    row = X_train.shape[0]
    X_train = np.concatenate((np.ones((row,1)), X_train), axis=1)

    if(mode == "b"):
        #strat = int(parameters[0])
        strat = 3
        #learning_rate = parameters[1]
        learning_rate  = ['2.5,0.1,0.5', '2.5, 0.2, 0.5', '2.5, 0.3, 0.5', '2.5, 0.4, 0.5', '2.5, 0.5, 0.5']
        rate  = ['0.1', '0.2', '0.3', '0.4', '0.5']
        #iterations = int(parameters[2])
        iterations = 250
        batches = 200
        #batches = [100,200,400,800,1600]
       
        for i in range(len(learning_rate)):
            global flag 
            flag = True
            Weight = np.zeros((X_train.shape[1], Y_train.shape[1]))
            name = output + "_" + rate[i] + ".csv"
            Weight = Weight_matrix_b(X_train, Y_train, Weight, strat, iterations, learning_rate[i], batches, name)
        


def adaptive(X_train, Y_train, param, mode, output):
    row = X_train.shape[0]
    X_train = np.concatenate((np.ones((row,1)), X_train), axis=1)
    strat = 2
    learning_rate = ['2','5','10','15','20']
    iterations = 250
    batches = 200
    
    for i in range(len(learning_rate)):
        global flag 
        flag = True
        Weight = np.zeros((X_train.shape[1], Y_train.shape[1]))
        name = output + "_" + learning_rate[i] + ".csv"
        Weight = Weight_matrix_b(X_train, Y_train, Weight, strat, iterations, learning_rate[i], batches, name)
    # np.savetxt(weight_file, Weight, delimiter="\n")

def batch_size(X_train, Y_train, param, mode, output):
    row = X_train.shape[0]
    X_train = np.concatenate((np.ones((row,1)), X_train), axis=1)
    f  = open(param, 'r')
    parameters = f.readlines()
    f.close()
    if(mode == "a"):
        strat = int(parameters[0])
        learning_rate = parameters[1]
        iterations = int(parameters[2])
    
        Weight = np.zeros((X_train.shape[1], Y_train.shape[1]))

        Weight = Weight_matrix(X_train, Y_train, Weight, strat, iterations, learning_rate, output)
        
        # np.savetxt(weight_file, Weight, delimiter="\n")

    if(mode == "b"):
        strat = int(parameters[0])
        # strat = 3
        learning_rate = parameters[1]
        #learning_rate  = ['2.5,0.1,0.5', '2.5, 0.2, 0.5', '2.5, 0.3, 0.5', '2.5, 0.4, 0.5', '2.5, 0.5, 0.5']
        #rate  = ['0.1', '0.2', '0.3', '0.4', '0.5']
        #iterations = int(parameters[2])
        iterations = 250
        #batches = 200
        batches = [100,200,400,800,1600]
       
        for i in range(len(learning_rate)):
            global flag 
            flag = True
            Weight = np.zeros((X_train.shape[1], Y_train.shape[1]))
            name = output + "_" + batches[i] + ".csv"
            Weight = Weight_matrix_b(X_train, Y_train, Weight, strat, iterations, learning_rate, batches[i], name)

def param_search(X_train, Y_train, param, mode):
    if(mode == "a"):
        for i in range(3):
            output = "Runa" + str(i+1) + ".csv"
            batch_size(X_train, Y_train, param[i], mode, output)
    if(mode == "b"):
        for i in range(3):
            output = "Runb"  
            batch_size(X_train, Y_train, param[i], mode, output)
    if(mode == "alpha"):
        output = "Runc"
        logistic_alpha(X_train, Y_train, param, mode)
    if(mode == "beta"):
        output = "Runc"
        logistic_beta(X_train, Y_train, param, mode)  
    if(mode == "adaptive"):
        output = "adap"
        adaptive(X_train, Y_train, param, mode, output)
# def find_param(X_train, Y_train, X_test):
def plot():
    for j in range(1):
        # X_a = []
        # Y_a = []
        output = "adap"
        #batches = [100,200,400,800,1600]
        batches = 200
        learning_rate = ['2','5','10','15','20']
        #Legend = ["100", "200", "400", "800", "1600"]
        for i in range(len(learning_rate)):
            name = output + "_" + learning_rate[i] + ".csv"
            df = pd.read_csv(name)
            y = df["Loss"]
            x = df["Runtime"]
            # X_a.append(x)
            # Y_a.append(y)

            plt.ylabel("Loss")
            plt.xlabel("Runtime (in secs)")
            plt.plot(x, y, marker = 'o')
        plt.legend(learning_rate)
        plt.title("adaptive_tuning")
        plt.savefig("adaptive" + '.png')
        plt.close()
            # plt2.ylabel("Loss")
            # plt2.xlabel("Runtime (in secs)")
            # plt2.plot(x, y, marker = 'o')
            # plt2.legend(["b" + str(j+1) + "_" + str(batches[i])])
            # plt2.savefig(name[:-4] + '.png')
            # plt2.close()
            

def check_plot():
    output = "Runb1_100.csv"
    df = pd.read_csv(output)
    y = df["Loss"]
    x = df["Runtime"]
    plt.ylabel("Loss")
    plt.xlabel("Runtime (in secs)")
    plt.plot(x,y,marker = 'o')
    plt.legend(['single element'])
    #plt.savefig(output[:-4] + '.png')
    plt.show()

# plot()

if __name__ == '__main__':
    mode = "adaptive"
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    param = ["param1.txt", "param2.txt", "param3.txt"]
    # output_file = sys.argv[5]
    # weight_file = sys.argv[6]
    
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
    
    Y_train = pd.get_dummies(Y_train)

    Y_train = Y_train.to_numpy()
    Y_train = Y_train.astype(float)


    param_search(X_train, Y_train, param, mode)

