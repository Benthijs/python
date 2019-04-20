

# Calculates the mean of a set of points
def mean(values):
    return sum(values) / float(len(values))


# multiplies the instance by its index as weight in the mean
def index_weighted_mean(x):
    xlist = []
    for i in range(len(x)):
        for a in range(x[i]):
            xlist.append(i)
    return int(sum(xlist) / len(xlist))


# Calculates the variance of a set of point
def variance(values, mean):
    return sum([(value - mean) ** 2 for value in values])


# Calculates the mean variance of a set of points
def antirelative_variance(values):
    m = mean(values)
    return mean([(value - m) ** 2 for value in values])


# Calculates the covariance of a set of 2 dimensional points
def covariance(x, x_m, y, y_m):
    return sum([(x[i] - x_m) * (y[i] - y_m) for i in range(len(x))])
