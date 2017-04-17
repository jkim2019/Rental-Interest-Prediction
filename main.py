import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy import stats
import random
import datetime as dt
import matplotlib.pyplot as plt

# set console width to meaningfully display data
DESIRED_WIDTH = 600
pd.set_option('display.width', DESIRED_WIDTH)
# set number of hidden layer units
HIDDEN_LAYER_SIZE = 15


def sigmoid(vector):
    '''
    :param vector: input vector, type ndarray
    :return: element-wise sigmoid of vector
    '''
    return np.power((1 + np.exp(-vector)), -1)

def nnCostFunction(X, y, lambda_value, theta_one, theta_two, num_labels):
    '''
    :param X: m x n matrix containing training data. include bias units
    :param y: m x 1 ndarray - objective value
    :param lambda_value: regularization parameter
    :param theta_one: weight matrix from input to hidden layer
    :param theta_two: weight matrix from hidden to output layer
    :param num_labels: number of nodes at the output layer
    :return: cost with given values
    '''


def costFunction(X, y, theta):
    '''
    :param X: m x n matrix - training data; includes bias units
    :param y: m x 1 ndarray - objective value
    :param theta: n x 1 ndarray - weights assigned with each attribute
    :return: array containing cost and gradient
    '''
    dim = X.shape
    m = dim[0]  # m: length
    n = dim[1]  # n: width

    cost = (1 / m) * (
    -y.transpose() * np.log(sigmoid(X * theta)) - (1 - y).transpose() * np.log(1 - sigmoid(X * theta)))


def main():
    # 1. data pre-processing (read JSON, print data set, add bias units, etc.)
    # 2. implement learning model
    # 4. (optional) plot X, y
    # 6. compute initial cost and gradient with initial theta
    # 7. use back-propagation algorithm to minimize theta
    # 8. recompute cost with new theta
    # 9a.(optional) plot data with decision boundaries
    # 10. predict interest values on test
    # 11. calculate error

    # 1.1 read JSON file. uses pandas (pd)
    print("reading JSON file ... \n\n")
    trainpd = pd.read_json("data/train.json")
    test = pd.read_json("data/test.json")

    # 1.2 translate JSON into numpy matrix
    # note: we lose attribute labels when performing this translation
    #   bathrooms(0)   bedrooms(1)   building_id(2)   created(3)   description(4)   display_address(5)   features(6)
    #   interest_level(7)   latitude(8)   listing_id(9)   longitude(10)   manager_id(11)   photos(12)   price(13)
    #   street_address(14)
    train_raw = trainpd.as_matrix()

    # 1.3 print some statistics
    # print("shape: \n\t{0}".format(trainpd.shape))
    print("info: \n{0}".format(trainpd.info))
    # print("description: \n{0}\n\n".format(trainpd.describe()))
    print(stats.describe(train_raw[:, 0]))

    print("\n\n")

    # 1.4 extract attributes to use in program
    # currently extracting: bathrooms, bedrooms, created, latitude, longitude, price
    train_x = train_raw[:, [0, 1, 3, 8, 10, 13]]
    train_y = train_raw[:, 7]
    dim = train_x.shape
    m = dim[0]  # m: length
    n = dim[1]  # n: width

    # 1.5 add bias unit
    ones_array = np.ones((m, 1))
    train_x = np.concatenate((ones_array, train_x), axis=1)
    n = n + 1 # adjust number of attributes to include bias unit
    print("adding bias unit")
    for element in train_x[0, :]:
        print(element)
    print("\n\n")

    # 1.6 translate created to age (use 2015-01-01 00:00:00 as reference frame)
    #     for algorithm needs, calculate only to hours
    for idx, date in enumerate(train_x[:, 3]):
        year = int(date[0:4])
        month = int(date[5:7])
        day = int(date[8:10])
        hour = int(date[11:13])
        minute = int(date[14:16])
        second = int(date[17:])
        cur_time = dt.datetime(year, month, day, hour, minute, second)
        ref_time = dt.datetime(2015, 1, 1, 0, 0, 0)
        time_diff_dt = cur_time - ref_time
        train_x[idx, 3] = time_diff_dt.total_seconds() // 3600

    # 1.7 translate train_y to numerical values
    for i in range(m):
        interest = train_y[i]
        if(interest == 'low'):
            train_y[i] = 0
        elif(interest == 'medium'):
            train_y[i] = 1
        elif(interest == 'high'):
            train_y[i] = 2
        else:
            print("ERROR: failed to translate y-value to integer")

    # 1.7.1 now we can cast train_x and train_y to type float
    train_x = train_x.astype(float)
    train_y = train_y.astype(float)

    print("translating date to age")
    for element in train_x[0, :]:
        print(element)
    print("\n\n")

    # 1.8 normalize values
    print("mean/std arrays (pre-processing")
    mean_array = np.mean(train_x, axis=0)
    print(mean_array)  # note: this includes bias unit
    std_array = np.std(train_x, axis=0)
    print(std_array)
    for i in range(1, n):
        train_x[:, i] = (train_x[:, i] - mean_array[i])/std_array[i]
    print("mean/std arrays (post-processing")
    mean_array = np.mean(train_x, axis=0)
    print(mean_array)  # note: this includes bias unit
    std_array = np.std(train_x, axis=0)
    print(std_array)
    print("\n\n")

    # 1.9 split training set into training set and cross-validation set
    #     use 10% of examples for cross-validation set. randomly selected rows
    cv_example_num = int(0.1*m)
    cv_set = np.zeros((cv_example_num, n), dtype=np.float)
    random_rows = random.sample(range(m), cv_example_num)
    random_rows = sorted(random_rows, reverse=True)
    for i in range(cv_example_num):
        cv_set[i] = train_x[random_rows[i], :]
        train_x = np.delete(train_x, (random_rows[i]), axis=0)

    print("train/cv dimensions")
    print(cv_set.shape)
    print(train_x.shape)
    print("\n\n")

    # 2.1 initialize values for nn setup
    input_layer_size = n - 1  # adjust for bias unit
    num_labels = 3
    theta_one = np.random.rand(HIDDEN_LAYER_SIZE, n)  # weight matrix from input to hidden layer
    theta_two = np.random.rand(3, HIDDEN_LAYER_SIZE + 1)  # weight matrix from hidden layer to output
    lambda_value = 0  # regularization parameter

    # 2.2 calculate initial cost



main()