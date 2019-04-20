import math
import numpy
import onedlinearregress
import matplotlib.pyplot as plt
shading_commonality = list()


# Converts a list of frequencies to list with each values presence relative
# to its frequency.
def treat_x(x):
    xlist = []
    for i in range(len(x)):
        for a in range(x[i]):
            xlist.append(i)
    return xlist


# Finds the standard deviation of the median for the probabilty of a
# particular shade.
def stdev(xlist):
    fsum = onedlinearregress.variance(xlist, onedlinearregress.mean(xlist))
    return math.sqrt(fsum / sum(xlist))


# standardize the image shadings to a scale of 0 to 1
def standardize(image):
    new_image = list()
    for a in range(len(image)):
        for i in range(len(image[0])):
            new_image.append(image[a][i] / 255.)
    return new_image


# Converts an image to grayscale (i.e removes the extra color dimensions)
# and propegates some image features.
def extract_features(image):
    final_image = numpy.empty_like(image)
    im = numpy.array(sorted(list(numpy.ndarray.flatten(image.copy())) +
                            list(range(0, 256))))
    im_len = len(im)
    improved_im = list()
    for i in range(256):
        used = list(numpy.where(im == i)[0])
        improved_im.append(used[0])
    for a in range(int(len(image))):
        for i in range(int(len(image[0]))):
            p = (improved_im[image[a][i]] / float(im_len)) * 255.
            final_image[a][i] = p
    return numpy.array(final_image)


# TODO Optimize entire function
# TODO Make perturbing possible with images taken in bad lighting.
# Converts image to grayscale attempting to propegate the written text
# features in the process.
def to_grayscale(image):
    global shading_commonality
    final_image = numpy.empty_like(image)
    im = numpy.array(sorted(list(numpy.ndarray.flatten(image.copy())) +
                            list(range(0, 256))))
    im_len = len(im)
    improved_im = list()
    for i in range(256):
        used = list(numpy.where(im == i)[0])
        improved_im.append(used[0])  # list of the index at which the first
        # reference is made to a particular shade
    shading_commonality = [improved_im[i] - improved_im[i - 1] for i in
                           range(1, len(improved_im))]
    # following is used to precalculate image shading such that the standard
    # deviation and mean of the shadings can be found
    pbim = numpy.empty_like(image)  # comensurate to percentileboostedimagemap
    for a in range(len(image)):
        for i in range(len(image[0])):
            percentile = improved_im[image[a][i]] / float(im_len)
            pbim[a][i] = (percentile * 255.)
    treated_shading_commonality = treat_x(shading_commonality)
    standard_deviation_mean = onedlinearregress.mean(
        treated_shading_commonality)
    standard_deviation = stdev(treated_shading_commonality)
    calced_variance = onedlinearregress.variance(treated_shading_commonality,
                                                 standard_deviation_mean) / \
        sum(treated_shading_commonality)
    standard_deviation *= calced_variance
    standard_deviation_mean -= standard_deviation
    for a in range(int(len(image))):
        for i in range(int(len(image[0]))):
            p = image[a][i]
            if(p < standard_deviation_mean):
                p *= ((0.25 * image[a][i]) / 255.)
            # TODO optimize grayscale boost
            final_image[a][i] = p
    return numpy.array(final_image)


# TODO optimize entire function as it is definitely noteceable in 4k images.
# Increase variance of image using given magnitude and gradient boosting.
def image_variance_art(image, magnitude):
    final_image = numpy.empty_like(image)
    for a in range(int(len(image))):
        for i in range(int(len(image[0]))):
            for c in range(len(image[0][0])):
                p = image[a][i][c] * ((magnitude * image[a][i][c]) / 255.0)
                final_image[a][i][c] = p
    return numpy.array(final_image)


# Retrieves a chunk out of a numpy array
def chunk(image, row, column, stroke):
    return numpy.array([[image[row + ro][column + col]
                         for col in range(-1 * (stroke / 2), stroke / 2)] for
                        ro in range(-1 * (stroke / 2), stroke / 2)])


# Looks at a pixel with a given perimeter and determines how to adjust it based
# on the average of all surounding pixels.
def blur(image, stroke=4):
    final_image = image.copy()
    # numpy.lib.pad(final_image, stroke / 2, 128)
    # TODO add padding for larger blurs
    for a in range(stroke, len(image) - stroke):
        for i in range(stroke, len(image[0]) - stroke):
            achunk = chunk(image, a, i, stroke)
            final_image[a][i] = numpy.sum(achunk) / achunk.size
    return final_image


# Calculates the factor by which the vertical axis has to dilate
def calc_strech(adjacent_len, servo_angle):
    hypotnuse_len = (math.sin(90) * adjacent_len) / math.sin(90 - servo_angle)
    return hypotnuse_len


# TODO Improve function to make use less empirical
# Returns the stretched image, mainly for fun
# Principle axis dictates principle axis where 1 is the larger and 0 smaller
def auto_image_dilator(image, servo_angle=89, principle_axis=0):
    stretched_image_len = calc_strech(image.shape[1], servo_angle)
    stretch_ratio = (stretched_image_len) / image.shape[1]
    print stretched_image_len
    print image.shape
    dilated_image = image.copy()
    # print image.shape[1], stretch_ratio
    numPerNew = stretched_image_len / (stretched_image_len - image.shape[1])
    n = 0
    for i in range(len(dilated_image)):
        if(i % numPerNew <= stretch_ratio):
            # print i
            inserted_arr = numpy.array([a for a in dilated_image[i]])
            # print list(inserted_arr)
            dilated_image = numpy.insert(dilated_image, i, inserted_arr,
                                         principle_axis)
            n += 1
    # print n, (stretched_image_len - image.shape[1])
    print dilated_image.shape, image.shape
    return dilated_image


# Principle axis variable dictates the axis being dilated 0 being smaller 1 being larger axis
def image_stretcher(image, stretch_ratio, principle_axis=0):
    dilated_image = image.copy()
    stretched_image_len = image.shape[principle_axis]*stretch_ratio
    numPerNew = int(image.shape[principle_axis] / (stretched_image_len - image.shape[principle_axis]))
    for i in range(len(dilated_image)):
        if(i % numPerNew == 0):
            inserted_arr = numpy.array([a for a in dilated_image[i]])
            dilated_image = numpy.insert(dilated_image, i, inserted_arr,
                                         principle_axis)
	i+=1
    return dilated_image


def imagestretcher2D(image, stretch_ratio, stretch_ratio1=-1):
	if stretch_ratio1==-1: stretch_ratio1=stretch_ratio
	image = image_stretcher(image, stretch_ratio)
	return image_stretcher(image, stretch_ratio1, 1)


# Creates and displays a graph of the probability of each shade of gray
# helpfull for when analyzing an images make-up.
def show():
    plt.plot(shading_commonality)
    plt.xlabel('Luminosity')
    plt.ylabel('Presence')
    plt.show()
