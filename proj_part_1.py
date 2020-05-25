import scipy.io
import math
import numpy
from sklearn.metrics import confusion_matrix

# Initial dataset
Numpyfile = scipy.io.loadmat('mnist_data.mat')

train_x = Numpyfile['trX']  # 784x12116
train_y = Numpyfile['trY']  # 12116x1
test_x = Numpyfile['tsX']  # 784x2000
test_y = Numpyfile['tsY']  # 2002x1

"""
    2-D feature vector N(mu, sigma^2)

"""

mean_7 = []
std_7 = []
mean_8 = []
std_8 = []

train_x7 = []
train_x8 = []
train_y7 = []
train_y8 = []
test_x7 = []
test_x8 = []
test_y7 = []
test_y8 = []

# Dividing training set into '7' and '8'
for i in range(len(train_y[0])):
    if train_y[0][i] == 0.0:
        train_x7.append(train_x[i])
        train_y7.append(train_y[0][i])
    else:
        train_x8.append(train_x[i])
        train_y8.append(train_y[0][i])

# Dividing test set into '7' and '8'
for i in range(len(test_y[0])):
    if test_y[0][i] == 0.0:
        test_x7.append(test_x[i])
        test_y7.append(test_y[0][i])
    else:
        test_x8.append(test_x[i])
        test_y8.append(test_y[0][i])

# mean and std for train data
for i in train_x7:
    mean_7.append(numpy.mean(i))
    std_7.append(numpy.std(i))

for i in train_x8:
    mean_8.append(numpy.mean(i))
    std_8.append(numpy.std(i))

tr_x7 = [list(a) for a in zip(mean_7, std_7)]
tr_x8 = [list(a) for a in zip(mean_8, std_8)]

tr_x7 = numpy.array(tr_x7).transpose().tolist()
tr_x8 = numpy.array(tr_x8).transpose().tolist()

mean_7 = []
std_7 = []
mean_8 = []
std_8 = []
# mean and std for test data
for i in test_x7:
    mean_7.append(numpy.mean(i))
    std_7.append(numpy.std(i))

for i in test_x8:
    mean_8.append(numpy.mean(i))
    std_8.append(numpy.std(i))

tst_x7 = [list(a) for a in zip(mean_7, std_7)]
tst_x8 = [list(a) for a in zip(mean_8, std_8)]

# finding mean and std for the new features obtained for the data
M_7 = []
STD_7 = []
M_8 = []
STD_8 = []

for i in tr_x7:
    M_7.append(numpy.mean(i))
    STD_7.append(numpy.std(i))
for i in tr_x8:
    M_8.append(numpy.mean(i))
    STD_8.append(numpy.std(i))

"""
    Naive Bayes algorithm
"""

# Likelihood probability of digit 7 and digit 8
# p(x | Ci) = (1 / sqrt(2pi)*sigma) * (exp(-(x - mu)^2 / 2sigma^2))

prior_7 = len(train_x7) / len(train_x)
prior_8 = len(train_x8) / len(train_x)

pred_Y_7 = []

for i in range(len(tst_x7)):
    prob_class_given_C_0 = []

    log_likelihood = 0
    for j in range(len(tst_x7[0])):
        try:
            r = - math.pow((tst_x7[i][j] - M_7[j]), 2) / (2 * math.pow(STD_7[j], 2))
            u = 1 / (math.sqrt(2 * math.pi) * STD_7[j])
            E = math.exp(r)
            v = math.log(u * E)
            log_likelihood = log_likelihood + v
        except(ZeroDivisionError):
            r = 0
            u = 0
        except(ValueError):
            v = 0
    prob_class_given_C_0.append(math.log(prior_7) + log_likelihood)

    prob_class_given_C_1 = []

    # Loop through every image
    log_likelihood = 0
    for j in range(len(tst_x7[0])):
        try:
            r = - math.pow((tst_x7[i][j] - M_8[j]), 2) / (2 * math.pow(STD_8[j], 2))
            u = 1 / (math.sqrt(2 * math.pi) * STD_8[j])
            E = math.exp(r)
            v = math.log(u * E)
            log_likelihood = log_likelihood + v
        except(ZeroDivisionError):
            r = 0
            u = 0
        except(ValueError):
            v = 0
    prob_class_given_C_1.append(math.log(prior_8) + log_likelihood)

    # comparing the predicted result of the test data with the target values of test data to get the accuracy
    if prob_class_given_C_0 > prob_class_given_C_1:
        pred_Y_7.append(0.0)
    else:
        pred_Y_7.append(1.0)

# calculating accuracy
acc = 0
for i in range(len(pred_Y_7)):
    if pred_Y_7[i] == test_y7[i]:
        acc += 1

acc = acc / len(pred_Y_7)
print("NAIVE BAYES CLASSIFIER:")
print("Accuracy for Naive Bayes for digit '7':", acc * 100, "%")

pred_Y_8 = []

for i in range(len(tst_x8)):
    prob_class_given_C_0 = []

    log_likelihood = 0
    for j in range(len(tst_x8[0])):
        try:
            r = - math.pow((tst_x8[i][j] - M_7[j]), 2) / (2 * math.pow(STD_7[j], 2))
            u = 1 / (math.sqrt(2 * math.pi) * STD_7[j])
            E = math.exp(r)
            v = math.log(u * E)
            log_likelihood = log_likelihood + v
        except(ZeroDivisionError):
            r = 0
            u = 0
        except(ValueError):
            v = 0
    prob_class_given_C_0.append(math.log(prior_7) + log_likelihood)

    prob_class_given_C_1 = []

    # Loop through every image
    log_likelihood = 0
    for j in range(len(tst_x8[0])):
        try:
            r = - math.pow((tst_x8[i][j] - M_8[j]), 2) / (2 * math.pow(STD_8[j], 2))
            u = 1 / (math.sqrt(2 * math.pi) * STD_8[j])
            E = math.exp(r)
            v = math.log(u * E)
            log_likelihood = log_likelihood + v
        except(ZeroDivisionError):
            r = 0
            u = 0
        except(ValueError):
            v = 0
    prob_class_given_C_1.append(math.log(prior_8) + log_likelihood)

    # comparing the predicted result of the test data with the target values of test data to get the accuracy

    if prob_class_given_C_0 > prob_class_given_C_1:
        pred_Y_8.append(0.0)
    else:
        pred_Y_8.append(1.0)

acc = 0
for i in range(len(pred_Y_8)):
    if pred_Y_8[i] == test_y8[i]:
        acc += 1
# calculating accuracy

acc = acc / len(pred_Y_8)

print("Accuracy for Naive Bayes for digit '8':", acc * 100, "%")
print("=========================================================")
print("LOGISTIC REGRESSION:")
"""
    Logistic Regression
"""
# converting train and test data into array
train_x7 = numpy.array(train_x7)
train_x8 = numpy.array(train_x8)
train_y7 = numpy.array(train_y7)
train_y8 = numpy.array(train_y8)
test_x7 = numpy.array(test_x7)
test_x8 = numpy.array(test_x8)


# sigmoid function
def sigmoid(scores):
    return 1 / (1 + numpy.exp(-scores))


# initialize the parameter: weights
weights = numpy.zeros(train_x7.shape[1])


# defining the log- likelihood function
def log_likelihood(train_x7, train_y7, weights):
    scores = numpy.dot(train_x7, weights)
    ll = numpy.sum(train_y7 * scores - numpy.log(1 + numpy.exp(scores)))
    return ll


# defining the main logistic regression function
def logistic_regression(train_x7, train_y7, weights, num_steps, learning_rate):
    init = log_likelihood(train_x7, train_y7, weights)
    print("initial ", init)

    for step in range(num_steps):
        scores = numpy.dot(train_x7, weights)  # calculate the score to be sent to sigmoid function
        predictions = sigmoid(scores)

        error = train_y7 - predictions
        gradient = numpy.dot(train_x7.transpose(), error)
        # updating the weights using gradient ascent
        weights += (learning_rate * gradient)

    if step % 10000 == 0:
        print(log_likelihood(train_x7, train_y7, weights))

    return weights


weights = logistic_regression(train_x7, train_y7, weights, 1000, 0.001)
final_scores_C1 = numpy.round(numpy.dot(weights.transpose(), test_x7.transpose()))
final_scores_C0 = numpy.round(-numpy.dot(weights.transpose(), test_x7.transpose()))
preds_7 = []

for i in range(len(final_scores_C0)):
    if final_scores_C0[i] > final_scores_C1[i]:
        preds_7.append(0.0)
    else:
        preds_7.append(1.0)

# calculating accuracy
acc = 0
for i in range(len(preds_7)):
    if preds_7[i] == test_y7[i]:
        acc += 1

acc = acc / len(preds_7)

print("Accuracy for Logistic Regression for digit '7':", acc * 100, "%")

# Calculating LR for digit 8
weights = numpy.zeros(train_x8.shape[1])


def log_likelihood(train_x8, train_y8, weights):
    scores = numpy.dot(train_x8, weights)
    ll = numpy.sum(train_y8 * scores - numpy.log(1 + numpy.exp(scores)))
    return ll


def logistic_regression(train_x8, train_y8, weights, num_steps, learning_rate):
    init = log_likelihood(train_x8, train_y8, weights)
    print("initial ", init)

    for step in range(num_steps):
        scores = numpy.dot(train_x8, weights)
        predictions = sigmoid(scores)

        error = train_y8 - predictions
        gradient = numpy.dot(train_x8.transpose(), error)
        # updating weights using gradient ascent
        weights += (learning_rate * gradient)

    if step % 10000 == 0:
        print(log_likelihood(train_x8, train_y8, weights))

    return weights


weights = logistic_regression(train_x8, train_y8, weights, 1000, 0.001)
final_scores_C1 = numpy.round(numpy.dot(weights.transpose(), test_x8.transpose()))
final_scores_C0 = numpy.round(-numpy.dot(weights.transpose(), test_x8.transpose()))
preds_8 = []

for i in range(len(final_scores_C0)):
    if final_scores_C0[i] > final_scores_C1[i]:
        preds_8.append(0.0)
    else:
        preds_8.append(1.0)
# calculating accuracy
acc = 0
for i in range(len(preds_8)):
    if preds_8[i] == test_y8[i]:
        acc += 1

acc = acc / len(preds_8)

print("Accuracy for Logistic Regression for digit '8':", acc * 100, "%")
