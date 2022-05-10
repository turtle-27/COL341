import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sys

def cross_entropy_loss(A_out, Y_train):
    eps=1e-15
    return abs(np.sum(np.log(np.clip(np.sum(A_out*Y_train, axis=1), eps, 1-eps)))/Y_train.shape[0])

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
def Momentum(W, delta_L, learning_rate, num_hidden, prev_delta, flag):
    gamma = 0.9
    for i in range(num_hidden+1):
        if(flag):
            W[i] = W[i] - learning_rate*delta_L[i]
            prev_delta.append(learning_rate*delta_L[i])
        else:
            v = gamma*prev_delta[i] + learning_rate*delta_L[i]
            W[i] = W[i] - v
            prev_delta[i] = v
    return W

def rmsProp(W, delta_L, learning_rate, num_hidden, prev_delta, flag):
    gamma = 0.9
    e = 1e-8
    for i in range(num_hidden+1):
        if(flag):
            prev_delta.append(0.1*np.square(delta_L[i]))
            W[i] = W[i] - (learning_rate/np.sqrt(prev_delta[i] + e))*delta_L[i]
        else:
            v = 0.9*prev_delta[i] + 0.1*np.square(delta_L[i])
            W[i] = W[i] - (learning_rate/np.sqrt(v+e))*delta_L[i]
            prev_delta[i] = v
    return W

def Adam(W, delta_L, learning_rate, num_hidden, prev_m, prev_v, t, flag):
    beta1 = 0.9
    beta2 = 0.99
    e = 1e-8
    for i in range(num_hidden+1):
        if(flag):
            prev_m.append((1-beta1)*delta_L[i])
            prev_v.append((1-beta2)*np.square(delta_L[i]))
            m = prev_m[i]/(1-beta1**t)
            v = prev_v[i]/(1-beta2**t)
            W[i] = W[i] - (learning_rate/(np.sqrt(v) + e))*m
        else:
            prev_m[i] = beta1*prev_m[i] + (1-beta1)*delta_L[i]
            prev_v[i] = beta2*prev_v[i] + (1-beta2)*np.square(delta_L[i])
            m = prev_m[i]/(1-beta1**t)
            v = prev_v[i]/(1-beta2**t)
            W[i] = W[i] - (learning_rate/(np.sqrt(v) + e))*m
    return W

def Nadam(W, delta_L, learning_rate, num_hidden, prev_m, prev_v, t, flag):
    beta1 = 0.9
    beta2 = 0.99
    e = 1e-8
    for i in range(num_hidden+1):
        if(flag):
            prev_m.append((1-beta1)*delta_L[i])
            prev_v.append((1-beta2)*np.square(delta_L[i]))
            m = prev_m[i]/(1-beta1**t)
            v = prev_v[i]/(1-beta2**t)
            W[i] = W[i] - (learning_rate/(np.sqrt(v) + e))*(beta1*m + ((1-beta1)*delta_L[i])/(1-beta1**t))
        else:
            prev_m[i] = beta1*prev_m[i] + (1-beta1)*delta_L[i]
            prev_v[i] = beta2*prev_v[i] + (1-beta2)*np.square(delta_L[i])
            m = prev_m[i]/(1-beta1**t)
            v = prev_v[i]/(1-beta2**t)
            W[i] = W[i] - (learning_rate/(np.sqrt(v) + e))*(beta1*m + ((1-beta1)*delta_L[i])/(1-beta1**t))
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
    # print(A[-1])
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

def plot(Y, plot_lr, plot_act, mode):
    X = []
    for i in range(len(Y)):
        X.append(i+1)
    plt.plot(X, Y, marker = 'o')
    plt.ylabel("CE Loss")
    plt.xlabel("Epochs")
    plt.title("momentum_arch_1")
    plt.savefig(mode+"param2_"+plot_lr+"_"+plot_act+".png")
    plt.close()

def main():
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    # param = sys.argv[3]
    
    train = pd.read_csv(input_path+"train_data_shuffled.csv", header = None)
    test = pd.read_csv(input_path+"public_test.csv", header = None)

    #Loading input    

    X_train = train.iloc[:, :-1]
    Y_train = train.iloc[:, -1]
    X_test = test.iloc[:, :-1]
    Y_true = test.iloc[:, -1].to_numpy()
    Y_train = pd.get_dummies(Y_train)    #one-hot encodding
    Y_train = Y_train.to_numpy()
    X_train = X_train.to_numpy()/255
    X_test = X_test.to_numpy()/255
    feature = X_train.shape[1]
    X_train = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), axis= 1)
    X_test = np.concatenate((np.ones((X_test.shape[0], 1)), X_test), axis= 1)


    epochs = 5
    batch_size = 100

    neurons = [512,256,128,64,46]
    learning_rate_type = 1
    learning_rate = 0.01
    activation_type = 1
    loss_type = 0
    seed_value = 1
    num_hidden = len(neurons) - 1
    learning_rate_types = [0,1]
    activation_types = [0,1,2]

    #Plotting param
    modes = ["gradient", "momentum", "nesterov" "adam", "nadam"]
    

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
    for mode in modes:
        for lr_rate in learning_rate_types:
            for act_type in activation_types:
                CE = []
                prev_delta = []
                prev_m = []
                prev_v = []
                flag = True
                for i in range(epochs):
                    for j in range(len(X_batch)):
                        if(mode == "gradient" or mode == "momentum"):
                            A = forward_propagation(X_batch[j], W, act_type, num_hidden, loss_type)
                            delta_L = backward_propagation(W,A,Y_batch[j],act_type, num_hidden, loss_type)
                            # break
                            if(mode == "gradient"):
                                W = Gradient(W, delta_L, learning_rate, num_hidden)
                            elif(mode == "momentum"):
                                W = Momentum(W, delta_L, learning_rate, num_hidden, prev_delta, flag)
                                flag = False
                        elif(mode == "nesterov"):
                            if(flag):
                                A = forward_propagation(X_batch[j], W, act_type, num_hidden, loss_type)
                            else:
                                temp = [W[k] - 0.9*prev_delta[k] for k in range(len(W))]
                                A = forward_propagation(X_batch[j], temp, act_type, num_hidden, loss_type)
                            delta_L = backward_propagation(W,A,Y_batch[j],act_type, num_hidden, loss_type)
                            # break
                            W = Momentum(W, delta_L, learning_rate, num_hidden, prev_delta, flag)
                            flag = False
                        elif(mode == "rmsProp"):
                            A = forward_propagation(X_batch[j], W, act_type, num_hidden, loss_type)
                            delta_L = backward_propagation(W,A,Y_batch[j],act_type, num_hidden, loss_type)
                            # break
                            W = rmsProp(W, delta_L, learning_rate, num_hidden, prev_delta, flag)
                            flag = False
                        elif(mode == "adam"):
                            A = forward_propagation(X_batch[j], W, act_type, num_hidden, loss_type)
                            delta_L = backward_propagation(W,A,Y_batch[j],act_type, num_hidden, loss_type)
                            # break
                            W = Adam(W, delta_L, learning_rate, num_hidden, prev_m, prev_v, j+1, flag)
                            flag = False
                        elif(mode == "nadam"):
                            A = forward_propagation(X_batch[j], W, act_type, num_hidden, loss_type)
                            delta_L = backward_propagation(W,A,Y_batch[j],act_type, num_hidden, loss_type)
                            # break
                            W = Nadam(W, delta_L, learning_rate, num_hidden, prev_m, prev_v, j+1, flag)
                            flag = False
                    A = forward_propagation(X_train, W, act_type, num_hidden, loss_type)
                    CE.append(cross_entropy_loss(A[-1], Y_train))
                    if(lr_rate == 1):
                        learning_rate = learning_rate/np.sqrt(i+1)

                
                if(lr_rate == 0):
                    plot_lr = "fixed_"+str(0.01)
                else:
                    plot_lr = "adap_"+str(0.01)
                if(act_type == 0):
                    plot_act = "sig"
                elif(act_type == 1):
                    plot_act = "tanh"
                else:
                    plot_act = "relu"
                plot(CE, plot_lr, plot_act, mode)
                np.savetxt(mode+"param2_"+plot_lr+"_"+plot_act+".csv", CE)
                # break
                
                
                #true_val = pd.read_csv("../toy_dataset/toy_dataset_test_labels.csv", header=None).to_numpy()
            










if __name__ == '__main__':
    main()