import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy import stats, optimize
import random
import datetime as dt

# set console width to meaningfully display data
DESIRED_WIDTH = 600
pd.set_option('display.width', DESIRED_WIDTH)

# set number of hidden layer units & classification labels
HIDDEN_LAYER_SIZE = 15
NUM_LABELS = 3

# set True to create new Test/Train sets
REORGANIZE_SETS = False

# set ratio of Test/Train sets. TEST_RATIO = x, Train = (1-x)
TEST_RATIO = 0.1

# regularization parameter
REG_PARAM = 1


def sigmoid(vector):
    """
    :param vector: input vector, type ndarray
    :return: element-wise sigmoid of vector
    """
    return 1 / (1 + np.exp(-vector))


def sigmoid_gradient(vector):
    """
    :param vector: input vector, type ndarray
    :return: gradient of sigmoid function at vector
    """
    return np.multiply(sigmoid(vector), (1 - sigmoid(vector)))


def forward_propagation(X, thetas):
    """
    :param X: m x n matrix containing bias units
    :param thetas: unrolled theta_one and theta_two
    :return: list containing z_two, alpha_two, z_three, h (hypothesis)
    """
    m, n = X.shape[0], X.shape[1]
    X = np.matrix(X)

    # reshape thetas
    theta_one = np.reshape(thetas[: HIDDEN_LAYER_SIZE * n], (HIDDEN_LAYER_SIZE, n))
    theta_two = np.reshape(thetas[HIDDEN_LAYER_SIZE * n:], (NUM_LABELS, HIDDEN_LAYER_SIZE + 1))

    # forward propagation
    z_two = X * theta_one.T
    alpha_two = np.insert(sigmoid(z_two), 0, values=np.ones(m), axis=1)  # take sigmoid(z_two) and add bias column
    z_three = alpha_two * theta_two.T
    h = sigmoid(z_three)

    return [z_two, alpha_two, z_three, h]


def calculate_cost(thetas, X, y, lambda_value, hl_size, num_l):
    """
    :param thetas: unraveled array containing theta_one and theta_two
    :param args: list containing X, y, lambda, hidden_layer_size, number_outputs
    :return: cost with regularization
    """
    X = np.matrix(X)
    m, n = X.shape[0], X.shape[1]

    # reshape thetas
    theta_one = np.reshape(thetas[: HIDDEN_LAYER_SIZE * n], (HIDDEN_LAYER_SIZE, n))
    theta_two = np.reshape(thetas[HIDDEN_LAYER_SIZE * n:], (NUM_LABELS, HIDDEN_LAYER_SIZE + 1))

    # perform forward propagation
    propagation = forward_propagation(X, thetas)
    z_two, alpha_two, z_three, h = propagation[0], propagation[1], propagation[2], propagation[3]

    # compute cost function as (m x 1) vector with costs associated w/ each example
    cost_vec = np.zeros((m, 1), dtype=float)

    # create y_mat as m x (NUM_LABELS). allows vector comparisons in cost_vec calculations
    identity = np.eye(NUM_LABELS, dtype=float)
    y_mat = np.zeros((m, NUM_LABELS), dtype=float)
    for i in range(m):
        y_mat[i] = identity[int(y[i]), :]

    for i in range(m):
        # noinspection PyTypeChecker
        cost_vec[i] = np.sum((-1) * (np.multiply(y_mat[i, :], np.log(h[i, :])) +
                                         np.multiply((1 - y_mat[i, :]), np.log(1 - h[i, :]))))
    cost_vec = (1 / m)*cost_vec
    # add regularization
    total_cost = np.sum(cost_vec)
    total_cost += (float(lambda_value) / (2 * m)) * (np.sum(np.power(theta_one[:, 1:], 2)) + np.sum(np.power(theta_two[:,
                                                                                             1:], 2)))
    return total_cost


def back_propagate(thetas, X, y, lambda_value, hl_size, num_l):
    """
    :param thetas: unraveled array containing theta_one and theta_two
    :param args: list containing X, y, lambda, hidden_layer_size, number_outputs
    :return: list: [cost, theta_grads]. NOTE: theta_grads is unravelled theta_grad_one & theta_grad_two
    """
    X = np.matrix(X)

    dim = X.shape
    m = dim[0]  # m: length
    n = dim[1]  # n: width

    theta_one = np.reshape(thetas[: hl_size * n], (hl_size, n))
    theta_two = np.reshape(thetas[hl_size * n:], (num_l, hl_size + 1))

    # values to calculate
    theta_one_grad = np.zeros(theta_one.shape)
    theta_two_grad = np.zeros(theta_two.shape)

    # perform forward propagation
    propagation = forward_propagation(X, thetas)
    z_two, alpha_two, z_three, h = propagation[0], propagation[1], propagation[2], propagation[3]

    # create y_mat as m x (NUM_LABELS). allows vector comparisons in cost_vec calculations
    y_vec = np.eye(NUM_LABELS, dtype=float)
    y_mat = np.zeros((m, NUM_LABELS), dtype=float)
    for i in range(m):
        y_mat[i] = y_vec[int(y[i]), :]

    # compute cost
    cost = calculate_cost(thetas, X, y, lambda_value, hl_size, num_l)

    for i in range(m):
        # forward propagation
        a1 = X[i, :]
        z_two_row = z_two[i, :]
        z_two_row = np.append([1], z_two_row)
        a_two_row = alpha_two[i, :]
        h_row = h[i, :]     # essentially a3. hypothesis/output

        # a2 = np.concatenate(([1], a_two_row)) no need... a2 already has bias unit

        # back propagation
        d3 = h_row - y_mat[i, :]
        d2 = np.multiply(d3 @ theta_two, sigmoid_gradient(z_two_row))

        # cumulative gradients
        theta_one_grad += d2[:, 1:].T @ a1
        theta_two_grad += d3.T @ a_two_row

    # divide by example num and regularize gradients
    theta_one_grad = (1 / m) * theta_one_grad + (lambda_value / m) * theta_one_grad
    theta_two_grad = (1 / m) * theta_two_grad + (lambda_value / m) * theta_two_grad
    theta_grads = np.append(theta_one_grad.ravel(), theta_two_grad.ravel())

    print("iteration")
    return cost, theta_grads


def prediction(h):
    """
    :param h: hypothesis vector
    :return: m x 1 prediction vector using argmax
    """
    m = h.shape[0]
    predicted = np.zeros((m, 1))
    for i in range(m):
        # obtain index with highest hypothesis value
        predicted[i] = np.argmax(h[i, :])
    return predicted


def accuracy(predicted, y):
    dim = predicted.shape
    m = dim[0]
    cur_sum = 0.0
    for i in range(m):
        if predicted[i] == y[i]:
            cur_sum += 1
    return cur_sum / float(m)


def main():
    """
    - this program creates a neural network, trains it on a JSON test set, and uses the trained model to make
    predictions on a JSON test set
    - test and train sets should be located in data/ as test.json and train.json respectively
    - on first run, enable REORGANIZE_SETS to read data
    """
    if REORGANIZE_SETS:
        # 1.1 read JSON file. uses pandas (pd)
        print("reading JSON file ... \n\n")
        trainpd = pd.read_json("data/train.json")

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
        n = n + 1  # adjust number of attributes to include bias unit
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
            if interest == 'low':
                train_y[i] = 0
            elif interest == 'medium':
                train_y[i] = 1
            elif interest == 'high':
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
        print("performing unity-based normalization")
        for i in range(1, n):
            max = np.max(train_x[:, i])
            min = np.min(train_x[:, i])
            train_x[:, i] = (train_x[:, i] - min) / (max - min)
        print("\n\n")

        # 1.9 split training set into training set and test set
        #     use 10% of examples for test set. randomly selected rows
        test_example_num = int(TEST_RATIO * m)
        test_set = np.zeros((test_example_num, n + 1), dtype=np.float)
        random_rows = random.sample(range(m), test_example_num)
        random_rows = sorted(random_rows, reverse=True)
        for i in range(test_example_num):
            test_set[i] = np.append(train_x[random_rows[i], :], train_y[random_rows[i]])
            train_y = np.delete(train_y, (random_rows[i]), axis=0)
            train_x = np.delete(train_x, (random_rows[i]), axis=0)
        np.save("test_set.npy", test_set)
        np.save("train_x.npy", train_x)
        np.save("train_y.npy", train_y)

    else:
        # otherwise, simply load saved data
        test_set = np.load("test_set.npy")
        train_x = np.load("train_x.npy")
        train_y = np.load("train_y.npy")

    dim = train_x.shape
    n = dim[1]  # n: width, includes bias unit. num_attributes = n - 1

    print("\n")
    print("test/train dimensions")
    print(test_set.shape)
    print(train_x.shape)
    print("\n")

    # 2.1 initialize values for nn setup
    # randomly initialize theta_one and theta_two
    thetas = (np.random.random(size=HIDDEN_LAYER_SIZE*n + NUM_LABELS*(HIDDEN_LAYER_SIZE + 1)) - 0.5)
    theta_one = np.reshape(thetas[: HIDDEN_LAYER_SIZE * n], (HIDDEN_LAYER_SIZE, n))
    theta_two = np.reshape(thetas[HIDDEN_LAYER_SIZE * n:], (NUM_LABELS, HIDDEN_LAYER_SIZE + 1))
    lambda_value = REG_PARAM  # regularization parameter

    # forward propagate once
    junk_1, junk_2, junk_3, h = forward_propagation(train_x, thetas)
    first_prediction = prediction(h)
    print("initial accuracy: {0}".format(accuracy(first_prediction, train_y)))
    print("\n")

    print("theta one/two dimensions")
    print(theta_one.shape)
    print(theta_two.shape)
    print("\n")

    # 2.2 calculate initial cost
    args = (train_x, train_y, lambda_value, HIDDEN_LAYER_SIZE, NUM_LABELS)
    cost = calculate_cost(thetas, train_x, train_y, lambda_value, HIDDEN_LAYER_SIZE, NUM_LABELS)
    print("cost with initial parameters: {0}".format(cost))

    # using conjugate gradient optimization method to optimize function
    optimized = optimize.minimize(back_propagate, x0=thetas, args=args, method='CG', jac=True, options={'maxiter': 10})
    print(optimized)

    # reshape theta
    theta_one = np.reshape(optimized.x[0: HIDDEN_LAYER_SIZE * n], (HIDDEN_LAYER_SIZE, n))
    theta_two = np.reshape(optimized.x[HIDDEN_LAYER_SIZE * n:], (NUM_LABELS, HIDDEN_LAYER_SIZE + 1))

    m, n = test_set.shape
    n = n - 1  # adjust for interest column

    # perform unity-based normalization
    for i in range(1, n):
        max = np.max(test_set[:, i])
        min = np.min(test_set[:, i])
        test_set[:, i] = (test_set[:, i] - min) / (max - min)

    # forward propagate one last time
    thetas = np.append(theta_one.ravel(), theta_two.ravel())
    junk_1, junk_2, junk_3, h = forward_propagation(test_set[:, 0:7], thetas)

    # test optimized theta values on new set
    second_prediction = prediction(h)
    print("second accuracy: {0}".format(accuracy(second_prediction, test_set[:, 7])))


main()
