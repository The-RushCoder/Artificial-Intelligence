#1
import pandas as pd
import math as m
import random
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + m.exp(-z))


def TrainingData(trainingDataset, lr):
    theta = [random.uniform(0, 1) for _ in range(len(trainingDataset[0]))]
    history = []
    itr = 0
    while itr <= 500:
        itr += 1
        TJ = 0
        for X in trainingDataset:
            z = sum([t * x for t, x in zip(theta, X[:-1] + [1])])
            if z > 100:
                h = 1
            elif z < -100:
                h = 0
            else:
                h = sigmoid(z)
            y = X[-1]
            h = min(max(h, 1e-15), 1 - 1e-15)
            J = -y * m.log(h) - (1 - y) * m.log(1 - h)
            TJ += J
            dv = [x * (h - y) for x in X[:-1] + [1]]
            theta = [t - (dvi * lr) for t, dvi in zip(theta, dv)]

        TJ /= len(trainingDataset)
        history.append(TJ)

    return theta, history


def Validation(theta):
    correct = 0
    for V in validationDataset:
        z = sum([t * v for t, v in zip(theta, V[:-1] + [1])])
        if z > 100:
            h = 1
        elif z < -100:
            h = 0
        else:
            h = sigmoid(z)
        if h >= 0.5:
            h = 1
        else:
            h = 0
        y = V[-1]
        if h == y:
            correct += 1
    val_acc = (correct * 100) / len(validationDataset)
    return val_acc


datacsv = pd.read_csv('dataset1.csv')
data_set = []
for index, row in datacsv.iterrows():
    item = row.values.tolist()
    data_set.append(item)

trainingDataset = data_set[:int(len(data_set) * .70)]
validationDataset = data_set[int(len(data_set) * .70):int(len(data_set) * .85)]
testDataset = data_set[-int(len(data_set) * .15):]

LearningRate = [0.1, 0.01, 0.001, 0.0001]
max_vRate = 0
for LR in LearningRate:
    theta, train_loss = TrainingData(trainingDataset, LR)
    vRate = Validation(theta)
    print("Learning Rate:", LR, "Validation Accuracy:", vRate)
    if vRate > max_vRate:
        Rate = LR
        max_vRate = vRate
        loss = train_loss
        bestTheta = theta

print("Max Validation Accuracy:", max_vRate, "For Learning Rate:", Rate)

# TESTING
correct = 0
for T in testDataset:
    z = sum([t * test for t, test in zip(bestTheta, T[:-1] + [1])])
    if z > 100:
        h = 1
    elif z < -100:
        h = 0
    else:
        h = sigmoid(z)
    if h >= 0.5:
        h = 1
    else:
        h = 0
    y = T[-1]
    if h == y:
        correct += 1
test_acc = (correct * 100) / len(testDataset)
print("Test Accuracy:", test_acc)

itr = [i for i in range(len(loss))]
plt.plot(itr, loss)
plt.xlabel('Epoch')
plt.ylabel('Train_loss')
plt.title('Epoch vs Train_loss')
plt.grid(True)
plt.show()





"""
#2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

#Reading a CSV file named "dataset.csv" into a pandas DataFrame called data_frame
data_frame = pd.read_csv("dataset.csv")
#Converting the DataFrame data_frame into a list of lists called dataset_list
dataset_list = data_frame.values.tolist()
#Initializing empty lists to store the training, validation, and test sets
Train_set = []
Val_set = []
Test_set = []

# Splitting dataset
#Looping through each row in the dataset list to split it into training, validation, and test sets based on random probabilities
for i in range(len(dataset_list)):
    x = random.uniform(0, 1)
    if 0 <= x <= 0.7:
        Train_set.append(dataset_list[i])
    elif 0.7 < x <= 0.85:
        Val_set.append(dataset_list[i])
    else:
        Test_set.append(dataset_list[i])

# Storing the value and removing the label
store_train = []
store_val = []
store_test = []

#Looping through each row in the training set to separate the label and features
for i in range(0, len(Train_set)):
    store_train.append(Train_set[i][0])
    del Train_set[i][0]
#Looping through each row in the validation set to separate the label and features
for i in range(0, len(Val_set)):
    store_val.append(Val_set[i][0])
    del Val_set[i][0]
#Looping through each row in the test set to separate the label and features
for i in range(0, len(Test_set)):
    store_test.append(Test_set[i][0])
    del Test_set[i][0]

# Inserting 1 to create X
#Looping through each row in the training set to add a bias term (1) to each feature vector
for i in range(len(Train_set)):
    Train_set[i].insert(0, 1)
#Looping through each row in the validation set to add a bias term (1) to each feature vector
for i in range(len(Val_set)):
    Val_set[i].insert(0, 1)
#Looping through each row in the test set to add a bias term (1) to each feature vector
for i in range(len(Test_set)):
    Test_set[i].insert(0, 1)

# Sigmoid Function
#Defining a sigmoid function to compute the sigmoid of a given input
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

# Logistic Regression

# Training
# Initializing the parameter vector Theta with random values
Theta = np.random.rand(3)
Train_loss = [] # Initializing an empty list to store training losses
Lr = 0.0001 #Setting the learning rate

#Looping through 1000 iterations for training the logistic regression model
for _ in range(1000):
    TJ = 0 #Calculating the average training loss
    #Looping through each sample in the training set to update the parameters using gradient descent
    for j in range(len(Train_set)):
        x = np.array(Train_set[j][:3])
        y = store_train[j]
        Z = np.dot(Theta, x)
        h = sigmoid(Z)
        # Clip the values to avoid math domain errors
        h = np.clip(h, 1e-15, 1 - 1e-15)
        J = (-y * np.log(h)) - ((1 - y) * np.log(1 - h))
        TJ += J
        dv = (h - y) * x
        Theta -= Lr * dv
    TJ /= len(Train_set)
    Train_loss.append(TJ) #Appending the average training loss to the list of training losses

# Graph
#Plotting the training loss over epochs
plt.plot(range(0, 1000), Train_loss)
plt.xlabel('epoch')
plt.ylabel('train-loss')
plt.show()

# Validation Set
Correct = 0
#Looping through each sample in the validation set to compute the validation accuracy
for i in range(len(Val_set)):
    x = np.array(Val_set[i][0:3])
    y = store_val[i]
    Z = np.dot(Theta, x)
    h = sigmoid(Z)
    if h >= 0.5:
        h = 1
    else:
        h = 0
    if h == y:
        Correct += 1

Validation_accuracy = (Correct / len(Val_set)) * 100 #Calculating the validation accuracy
print(f"The validation accuracy of the dataset is: {Validation_accuracy} %")

# Test Set
Correct = 0
#Looping through each sample in the test set to compute the test accuracy
for i in range(len(Test_set)):
    x = np.array(Test_set[i][0:3])
    y = store_test[i]
    Z = np.dot(Theta, x)
    h = sigmoid(Z)
    if h >= 0.5:
        h = 1
    else:
        h = 0
    if h == y:
        Correct += 1

Test_accuracy = (Correct / len(Test_set)) * 100 #Calculating the test accuracy
print(f"The Test accuracy of the dataset is: {Test_accuracy} %")
"""
