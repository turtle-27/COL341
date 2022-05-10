import pandas as pd
import numpy as np
import sys

def cross_entropy_loss():
    print("")

def softmax(x):
    x = np.exp(x)
    return x/(np.sum(x, axis = 1, keepdims= True))


def sigmoid(A):
    return 1/(1 + np.exp(-A))

def tanh(X):
    return np.tanh(X)

def relu(X):
    return np.maximum(0,X)

def Gradient(W, delta_L, learning_rate, num_hidden):
    for i in range(num_hidden+1):
        #print(i)
        W[i] = W[i] - learning_rate*delta_L[i]
    return W


def forward_propagation(X_train, W, activation_type, num_hidden, loss_type):
    A = []
    A.append(X_train)
    #A.append(np.concatenate((np.ones((X_train.shape[0], 1)), X_train), axis= 1))
    for i in range(1, num_hidden+2):
        #A[i-1] = np.concatenate((np.ones((A[i-1].shape[0], 1)), A[i-1]), axis= 1)
        z = A[i-1]@W[i-1]
        if(i != num_hidden+1):
            if(activation_type == 0):
                S = sigmoid(z)          
            elif(activation_type == 1):
                S = tanh(z)
            else:
                S = relu(z)
            S = np.concatenate((np.ones((S.shape[0], 1)), S), axis= 1)
        else:
            if(loss_type == 0):
                S = softmax(z)
            else:
                if(activation_type == 0):
                    S = sigmoid(z)          
                elif(activation_type == 1):
                    S = tanh(z)
                else:
                    S = relu(z)
        A.append(S)
    #print(A[-1])
    return A

def act_derivative(a, activation_type):
    if(activation_type == 0):
        return a*(1-a)
    elif(activation_type == 1):
        return (1-a*a)
    else:
        return ((a>0)).astype(int)


def backward_propagation(W, A, Y_train, activation_type, num_hidden, loss_type):
    delta_L = [0.0 for i in range(num_hidden+1)]
    #dC_da_i = (A[-1] - Y_train)/Y_train.shape[0]
    if(loss_type == 0):
        dC_da_i = (A[-1] - Y_train)/Y_train.shape[0]
    else:
        dC_da_i = ((A[-1] - Y_train)*act_derivative(A[-1], activation_type))/Y_train.shape[0]
    delta_L[-1] = np.dot(A[-2].T, dC_da_i)
    for i in range(num_hidden-1, -1, -1):
        dC_da_i = np.dot(dC_da_i, W[i+1].T)*act_derivative(A[i+1], activation_type)
        dC_da_i = np.delete(dC_da_i, 0, axis=1)
        delta_L[i] = np.dot(A[i].T, dC_da_i)
    return delta_L

def check(prediction, true_value):
    c = 0
    for i in range(len(prediction)):
        if(prediction[i] == true_value[i]):
            c+=1
    return c/len(prediction)



def main():
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    param = sys.argv[3]
    
    train = pd.read_csv(input_path+"toy_dataset_train.csv", header = None)
    test = pd.read_csv(input_path+"toy_dataset_test.csv", header = None)

    #Loading input    

    X_train = train.iloc[:, 1:]
    Y_train = train.iloc[:, 0]
    X_test = test.iloc[:, 1:]
    Y_train = pd.get_dummies(Y_train)    #one-hot encodding
    Y_train = Y_train.to_numpy()
    X_train = X_train.to_numpy()/255
    X_test = X_test.to_numpy()/255
    feature = X_train.shape[1]
    X_train = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), axis= 1)
    X_test = np.concatenate((np.ones((X_test.shape[0], 1)), X_test), axis= 1)


    f = open(param, 'r')
    lines = f.readlines()
    epochs = int(lines[0])
    batch_size = int(lines[1])
    temp = lines[2].strip('[]\n').split(',')
    neurons = [int(elem) for elem in temp]
    learning_rate_type = int(lines[3])
    learning_rate = float(lines[4])
    activation_type = int(lines[5])
    loss_type = int(lines[6])
    seed_value = int(lines[7])
    num_hidden = len(neurons) - 1

    #initializing weights and bias
    W = []
    np.random.seed(seed_value)
    weight = np.float32(np.random.normal(0, 1, size=(feature+1, neurons[0]))*np.sqrt(2/(feature+1+neurons[0])))
    weight = np.float64(weight)
    W.append(weight)
    for i in range(1,num_hidden+1):
        weight = np.float32(np.random.normal(0, 1, size=(neurons[i-1]+1, neurons[i]))*np.sqrt(2/(neurons[i-1] + 1 + neurons[i])))
        weight = np.float64(weight)
        W.append(weight)
    # print(W)  

    #mini batch gradient
    X_batch = []
    Y_batch = []
    no_batch = X_train.shape[0]//batch_size
    for i in range(no_batch):
        tempX = X_train[i*batch_size:(i+1)*batch_size, :]
        tempY = Y_train[i*batch_size:(i+1)*batch_size]
        X_batch.append(tempX)
        Y_batch.append(tempY)
    
    for i in range(epochs):
        for j in range(len(X_batch)):
            A = forward_propagation(X_batch[j], W, activation_type, num_hidden, loss_type)
            delta_L = backward_propagation(W,A,Y_batch[j],activation_type, num_hidden, loss_type)
            W = Gradient(W, delta_L, learning_rate, num_hidden)
        if(learning_rate_type == 1):
            learning_rate = learning_rate/np.sqrt(i+1)
    
    A = forward_propagation(X_test, W, activation_type, num_hidden, loss_type)
    prediction = np.argmax(A[-1], axis=1)
    
    # true_val = pd.read_csv("../toy_dataset/toy_dataset_test_labels.csv", header=None).to_numpy()
    # print(check(prediction, true_val))
    np.save(output_path+"predictions.npy", prediction)

    for i in range(num_hidden+1):
        np.save(output_path+"w_"+str(i+1)+".npy", W[i])
    










if __name__ == '__main__':
    main()
