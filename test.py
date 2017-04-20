import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy import optimize
import random
import datetime as dt

# set console width to meaningfully display data
DESIRED_WIDTH = 600
pd.set_option('display.width', DESIRED_WIDTH)

# set number of hidden layer units & classification labels
HIDDEN_LAYER_SIZE = 4
NUM_LABELS = 3

# set True to create new Test/Train sets
REORGANIZE_SETS = False

# set ratio of Test/Train sets. TEST_RATIO = x, Train = (1-x)
TEST_RATIO = 0.1

# regularization parameter
REG_PARAM = 0


def sigmoid(vector):
    """
    :param vector: input
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
    :param X: m x n matrix without bias units
    :param thetas: unrolled theta_one and theta_two
    :return: list containing z_two, alpha_two, z_three, h (hypothesis)
    """
    m, n = X.shape[0], X.shape[1]

    # reshape thetas
    theta_one = np.reshape(thetas[: HIDDEN_LAYER_SIZE * (n + 1)], (HIDDEN_LAYER_SIZE, (n + 1)))
    theta_two = np.reshape(thetas[HIDDEN_LAYER_SIZE * (n + 1):], (NUM_LABELS, HIDDEN_LAYER_SIZE + 1))

    # forward propagation
    alpha_one = np.insert(X, 0, values=np.ones(m), axis=1)
    z_two = np.matmul(alpha_one, theta_one.T)
    alpha_two = np.insert(sigmoid(z_two), 0, values=np.ones(m), axis=1)  # take sigmoid(z_two) and add bias column
    z_three = np.matmul(alpha_two, theta_two.T)
    h = sigmoid(z_three)

    return z_two, alpha_two, z_three, h


def calculate_cost(thetas, X, y, lambda_value, hl_size, num_l):
    """
    :param thetas: unraveled array containing theta_one and theta_two
    :param X: m x n example matrix without bias units
    :param y: interest level, expressed as 0 (low) to 2 (high)
    :param lambda_value: regularization parameter. although global, must be passed since func used in minimization
    :param hl_size: hidden layer node count (not including bias)
    :param num_l: number of outputs
    :return: cost
    """
    m, n = X.shape[0], X.shape[1]

    # reshape thetas
    theta_one = np.reshape(thetas[: hl_size * (n + 1)], (hl_size, (n + 1)))
    theta_two = np.reshape(thetas[hl_size * (n + 1):], (num_l, hl_size + 1))

    # forward propagation
    z_two, alpha_two, z_three, h = forward_propagation(X, thetas)

    cost_vec = np.zeros((m, 1))

    # use one-hot encoding
    identity = np.eye(num_l, dtype=float)
    y_mat = np.zeros((m, num_l), dtype=float)
    for i in range(m):
        y_mat[i] = identity[int(y[i]), :]

    for i in range(m):
        index_cost = (-1) * (np.multiply(y_mat[i, :], np.log(h[i, :])) +
                             np.multiply((1 - y_mat[i, :]), np.log(1 - h[i, :])))
        cost_vec[i] = np.sum(index_cost)

    total_cost = np.sum(cost_vec)
    total_cost = (1 / m) * total_cost
    # add regularization
    total_cost += (float(lambda_value) / (2 * m)) * (
        np.sum(np.power(theta_one[:, 1:], 2)) + np.sum(np.power(theta_two[:,
                                                                1:], 2)))
    return total_cost


def back_propagate(thetas, X, y, lambda_value, hl_size, num_l):
    """
    :param thetas: unraveled array containing theta_one and theta_two
    :param X: m x n example matrix without bias units
    :param y: interest level, expressed as 0 (low) to 2 (high)
    :param lambda_value: regularization parameter. although global, must be passed since func used in minimization
    :param hl_size: hidden layer node count (not including bias)
    :param num_l: number of outputs
    :return: cost, theta_grads containing unrolled theta_one_grad and theta_two_grad
    """
    dim = X.shape
    m = dim[0]  # m: length
    n = dim[1]  # n: width

    theta_one = np.reshape(thetas[: hl_size * (n + 1)], (hl_size, n + 1))
    theta_two = np.reshape(thetas[hl_size * (n + 1):], (num_l, hl_size + 1))

    # initializations
    theta_one_grad = np.zeros(theta_one.shape)
    theta_two_grad = np.zeros(theta_two.shape)

    # perform forward propagation
    propagation = forward_propagation(X, thetas)
    z_two, alpha_two, z_three, h = propagation[0], propagation[1], propagation[2], propagation[3]

    # create y_mat as m x (NUM_LABELS) (one-hot encoding)
    y_vec = np.eye(NUM_LABELS, dtype=float)
    y_mat = np.zeros((m, NUM_LABELS), dtype=float)
    for i in range(m):
        y_mat[i] = y_vec[int(y[i]), :]

    # compute cost
    # cost = calculate_cost(thetas, X, y, lambda_value, hl_size, num_l)
    # OPTIMIZATION: forward propagation already performed. calculate_cost performs
    # forward propagation again. instead, simply use code calculation portion of code
    cost_vec = np.zeros((m, 1))
    for i in range(m):
        index_cost = (-1) * (np.multiply(y_mat[i, :], np.log(h[i, :])) +
                             np.multiply((1 - y_mat[i, :]), np.log(1 - h[i, :])))
        cost_vec[i] = np.sum(index_cost)

    total_cost = np.sum(cost_vec)
    total_cost = (1 / m) * total_cost
    # add regularization
    total_cost += (float(lambda_value) / (2 * m)) * (
        np.sum(np.power(theta_one[:, 1:], 2)) + np.sum(np.power(theta_two[:,
                                                                1:], 2)))

    for i in range(m):
        # forward propagation
        a1 = np.append([1], X[i, :])  # add bias unit
        z_two_row = z_two[i, :]
        z_two_row = np.append([1], z_two_row)
        a_two_row = alpha_two[i, :]
        h_row = h[i, :]     # essentially a3. hypothesis/output

        # a2 = np.concatenate(([1], a_two_row)) no need... a2 already has bias unit

        # back propagation
        d3 = h_row - y_mat[i, :]
        d2 = np.multiply(d3 @ theta_two, sigmoid_gradient(z_two_row))

        # cumulative gradients
        a_two_row = np.matrix(a_two_row)
        theta_one_grad += np.matrix(d2)[:, 1:].T @ np.matrix(a1)
        theta_two_grad += np.matrix(d3).T @ np.matrix(a_two_row)

    # divide by example num and regularize gradients
    theta_one_grad = (1 / m) * theta_one_grad + (lambda_value / m) * theta_one_grad
    theta_two_grad = (1 / m) * theta_two_grad + (lambda_value / m) * theta_two_grad
    theta_grads = np.append(theta_one_grad.ravel(), theta_two_grad.ravel())

    return total_cost, theta_grads


def predict(h):
    """
    :param h: m x (NUM_LABELS) hypothesis matrix 
    :return: m x 1 prediction vector using argmax
    """
    m = h.shape[0]
    prediction = np.zeros((m, 1))
    for i in range(m):
        # obtain index with highest hypothesis value
        prediction[i] = np.argmax(h[i, :])
    return prediction


def accuracy(prediction, y):
    """
    :param prediction: m x 1 prediction vector
    :param y: interest level
    :return: accuracy, precision, recall
    """
    dim = prediction.shape
    m = dim[0]
    cur_sum = 0.0
    true_positives = [0.0, 0.0, 0.0]
    predicted_as = [0.0, 0.0, 0.0]
    actually_is = [0.0, 0.0, 0.0]
    precision = [0.0, 0.0, 0.0]
    recall = [0.0, 0.0, 0.0]
    for i in range(m):
        if prediction[i] == y[i]:
            cur_sum += 1

    for i in range(m):
        # j represents interest level
        predicted_as[int(np.around(prediction[i]))] += 1
        actually_is[int(np.around(y[i]))] += 1

        if int(np.around(prediction[i])) == int(np.around(y[i])):
            true_positives[int(np.around(prediction[i]))] += 1

    for i in range(3):
        if predicted_as[i] == 0:
            precision[i] = None
        else:
            precision[i] = true_positives[i] / predicted_as[i]
        if actually_is[i] == 0:
            recall[i] = None
        else:
            recall[i] = true_positives[i] / actually_is[i]

    return cur_sum / float(m), precision, recall


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

        # 1.3 extract attributes to use in program
        # currently extracting: bathrooms, bedrooms, created, latitude, longitude, price
        train_x = train_raw[:, [0, 1, 3, 8, 10, 13]]
        train_y = train_raw[:, 7]
        dim = train_x.shape
        m = dim[0]  # m: length
        n = dim[1]  # n: width

        # 1.4 translate created to age (use 2015-01-01 00:00:00 as reference frame)
        #     for algorithm needs, calculate only to hours
        for idx, date in enumerate(train_x[:, 2]):
            year = int(date[0:4])
            month = int(date[5:7])
            day = int(date[8:10])
            hour = int(date[11:13])
            minute = int(date[14:16])
            second = int(date[17:])
            cur_time = dt.datetime(year, month, day, hour, minute, second)
            ref_time = dt.datetime(2015, 1, 1, 0, 0, 0)
            time_diff_dt = cur_time - ref_time
            train_x[idx, 2] = time_diff_dt.total_seconds() // 3600

        # 1.5 translate train_y to numerical values
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

        # 1.5 now we can cast train_x and train_y to type float
        train_x = train_x.astype(float)
        train_y = train_y.astype(float)

        # 1.6 normalize values
        print("performing unity-based normalization")
        for i in range(0, n):
            max = np.max(train_x[:, i])
            min = np.min(train_x[:, i])
            train_x[:, i] = (train_x[:, i] - min) / (max - min)
        print("\n\n")

        # 1.7 split training set into training set and test set. randomly selected rows
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

        print("finished reorganizing sets\n\n")

    else:
        # otherwise, simply load saved data
        test_set = np.load("test_set.npy")
        train_x = np.load("train_x.npy")
        train_y = np.load("train_y.npy")

    # adjust skewed classes
    low_count, med_count, high_count = 0, 0, 0
    equal_x = np.zeros((10000, train_x.shape[1]))
    equal_y = np.zeros((10000, 1))
    counter = 0

    while low_count < 4000:
        if train_y[counter] == 0:
            equal_x[low_count, :] = train_x[counter, :]
            equal_y[low_count] = train_y[counter]
            low_count += 1
        counter += 1
    counter = 0
    while med_count < 3000:
        if train_y[counter] == 1:
            equal_x[med_count, :] = train_x[counter, :]
            equal_y[med_count] = train_y[counter]
            med_count += 1
        counter += 1
    counter = 0
    while high_count < 3000:
        if train_y[counter] == 2:
            equal_x[high_count, :] = train_x[counter, :]
            equal_y[high_count] = train_y[counter]
            high_count += 1
        counter += 1
    train_x = equal_x
    train_y = equal_y

    dim = train_x.shape
    m, n = dim[0], dim[1]   # m: # examples
                            # n: width, no bias units

    # 2.1 initialize values for nn setup
    # randomly initialize theta_one and theta_two
    thetas = (np.random.random(size=HIDDEN_LAYER_SIZE * (n + 1) + NUM_LABELS * (HIDDEN_LAYER_SIZE + 1)) - 0.5)
    lambda_value = REG_PARAM  # regularization parameter

    # forward propagate once
    junk_1, junk_2, junk_3, h = forward_propagation(train_x, thetas)
    first_prediction = predict(h)
    acc, precision, recall = accuracy(first_prediction, train_y)
    print("initial accuracy: {0}\ninitial precision: {1}\ninitial recall: {2}".format(acc, precision, recall))
    print("\n")

    # print("theta one/two dimensions")
    # print(theta_one.shape)
    # print(theta_two.shape)
    # print("\n")

    # 2.2 calculate initial cost
    args = (train_x, train_y, lambda_value, HIDDEN_LAYER_SIZE, NUM_LABELS)
    cost = calculate_cost(thetas, train_x, train_y, lambda_value, HIDDEN_LAYER_SIZE, NUM_LABELS)
    print("cost with initial parameters: {0}".format(cost))

    callback_cost = lambda cur_theta: print(calculate_cost(cur_theta, train_x, train_y, lambda_value, HIDDEN_LAYER_SIZE, NUM_LABELS))

    # using conjugate gradient optimization method to optimize function
    # jac = True allows algorithm to accept both cost and gradient from back_propagate
    cg_optimum = optimize.minimize(back_propagate, x0=thetas, args=args, method='CG', jac=True, options={'maxiter': 50}, callback=callback_cost)
    print(cg_optimum)

    # reshape theta
    theta_one = np.reshape(cg_optimum.x[0: HIDDEN_LAYER_SIZE * (n + 1)], (HIDDEN_LAYER_SIZE, n + 1))
    theta_two = np.reshape(cg_optimum.x[HIDDEN_LAYER_SIZE * (n + 1):], (NUM_LABELS, HIDDEN_LAYER_SIZE + 1))

    m, n = test_set.shape

    # perform unity-based normalization on test set
    for i in range(0, n - 1):
        max = np.max(test_set[:, i])
        min = np.min(test_set[:, i])
        test_set[:, i] = (test_set[:, i] - min) / (max - min)

    # forward propagate one last time
    thetas = np.append(theta_one.ravel(), theta_two.ravel())
    junk_1, junk_2, junk_3, h = forward_propagation(test_set[:, 0:n - 1], thetas)

    for i in range(30):
        print(h[i])

    # test optimized theta values on new set
    second_prediction = predict(h)

    for i in range(30):
        print(second_prediction[i])

    acc, precision, recall = accuracy(second_prediction, test_set[:, n - 1])
    print("final accuracy: {0}\nfinal precision: {1}\nfinal recall: {2}".format(acc, precision, recall))

main()