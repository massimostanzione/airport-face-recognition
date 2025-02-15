import math

from DES_Python.rvms import idfStudent


# welford
def estimate(valuesArray):
    if len(valuesArray) == 0:
        mean = 0.0
        stdev = 0.0
        w = 0.0
        return mean, stdev, w

    LOC = 0.95
    n = 0
    sum = 0.0
    mean = 0.0

    data = valuesArray[n]

    for i in range(1, len(valuesArray)):
        n += 1
        diff = float(data) - mean
        sum += diff * diff * (n - 1.0) / n
        mean += diff / n
        data = valuesArray[i]

    stdev = math.sqrt(sum / n)

    if n > 1:
        u = 1.0 - 0.5 * (1.0 - LOC)
        t = idfStudent(n - 1, u)
        w = t * stdev / math.sqrt(n - 1)

        return mean, stdev, w
    else:
        print("ERROR - insufficient data\n")

