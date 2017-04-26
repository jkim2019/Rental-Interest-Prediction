from main import *
import csv

def main():
    testpd = pd.read_json("data/test.json")
    thetas = np.load('thetas.npy')
    test_raw = testpd.as_matrix()

    test_x = test_raw[:, [0, 1, 3, 7, 9, 12]]
    dim = test_x.shape
    m = dim[0]  # m: length
    n = dim[1]  # n: width

    for idx, date in enumerate(test_x[:, 2]):
        year = int(date[0:4])
        month = int(date[5:7])
        day = int(date[8:10])
        hour = int(date[11:13])
        minute = int(date[14:16])
        second = int(date[17:])
        cur_time = dt.datetime(year, month, day, hour, minute, second)
        ref_time = dt.datetime(2015, 1, 1, 0, 0, 0)
        time_diff_dt = cur_time - ref_time
        test_x[idx, 2] = time_diff_dt.total_seconds() // 3600

    test_x = test_x.astype(float)

    print("performing unity-based normalization")
    for i in range(0, n):
        max = np.max(test_x[:, i])
        min = np.min(test_x[:, i])
        test_x[:, i] = (test_x[:, i] - min) / (max - min)
    print("\n\n")

    print(testpd.head())

    j1, j2, j3, j4, j5, h = forward_propagation(test_x[:, :], thetas)

    h = np.insert(h, 0, values=test_raw[:, 8], axis=1)

    myfile = open('submission.csv', 'w')
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    for i in range(m):
        wr.writerow(h[i, :])
main()
