import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

file_handler = open("Social_Network_Ads.csv", "r")
data = pd.read_csv(file_handler, sep=",", usecols=[1, 2, 3, 4])
file_handler.close()

gender = {'Male': 0, 'Female': 1}
data.Gender = [gender[item] for item in data.Gender]

df = pd.DataFrame(data)
df['Age'] = df['Age'].astype(float)
df['EstimatedSalary'] = df['EstimatedSalary'].astype(float)

data = np.array(data)
xdata = np.array(data[:, :-1])
ydata = np.array(data[:, -1])
ydata = ydata.astype(int)

train_x, test_x, train_y, test_y = train_test_split(xdata, ydata, test_size=0.25, random_state=0)

# x_copy = train_x.copy()
cols = len(train_x[0])

# def feature_scaling():
#    for i in range(cols):
#        maxx = np.max(train_x[:, i])
#        train_x[:, i] = train_x[:, i] / maxx
#        maxx = np.max(test_x[:, i])
#        test_x[:, i] = test_x[:, i] / maxx


# feature_scaling()

sc = StandardScaler()
train_x = sc.fit_transform(train_x)
test_x = sc.transform(test_x)

print(train_x)

weights = []
for p in range(cols):
    weights.append(0.)
print(weights)
bias = 0


def sigmoid(w, x, b):
    z = np.dot(x, w) + b
    return 1 / (1 + np.exp(-z))


def deriv_w(w, b, j, lam):
    return (np.dot(sigmoid(w, train_x, b) - train_y, train_x[:, j]) / len(train_x)) + (lam * w[j] / cols)


def deriv_b(w, b):
    return np.sum(sigmoid(w, train_x, b) - train_y) / len(train_x)


print(deriv_w(weights, 0, 0, 10), deriv_b(weights, 0))


def gradient_descent(w, b, epoch, alpha, lam):
    for i in range(epoch):
        temp = w
        for j in range(cols):
            w[j] = w[j] - alpha * deriv_w(w, b, j, lam)
        b = b - alpha * deriv_b(temp, b)
        print(w, b)
    return w, b


model_w, model_b = gradient_descent(weights, 0, 10000, 1, 0)


def predict(features, w, b, threshold):
    val = sigmoid(w, features, b)
    if val > threshold:
        return 1
    else:
        return 0


# def feature_scale_f(features):
#    for i in range(0, cols):
#        maxx = np.max(x_copy[:, i])
#        features[i] = features[i] / maxx
#    return features


# while True:
    # gender = float(input("Gender (0/1): "))
    # age = float(input("Age: "))
    # salary = float(input("Salary: "))
    # f = np.array([gender, age, salary])
    # f = np.array([age, salary])
    # f = feature_scale_f(f)
    # f = sc.fit_transform(f)
    # print(predict(f, model_w, model_b, 0.5))

y_hat = []


def test_model():
    for i in range(len(test_y)):
        y_hat.append(predict(test_x[i], model_w, model_b, 0.5))


def find_accuracy():
    total = 0
    for i in range(len(test_y)):
        print("y_hat:", y_hat[i], "y:", test_y[i])
        if y_hat[i] == test_y[i]:
            total += 1
    return (total / len(test_y)) * 100.0


test_model()
print("Model Accuracy: {}%".format(find_accuracy()))


new_linspace = np.linspace(data[0][0], data[-1][0])
plt.rcParams["figure.figsize"] = [8, 6]
plt.rcParams["figure.autolayout"] = True
value_color = {0: 'blue', 1: 'red'}
for p in range(len(train_x)):
    plt.scatter(train_x[p, 1], train_x[p, 2], color=value_color.get(train_y[p]))
plt.show()
