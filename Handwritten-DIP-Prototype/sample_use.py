import numpy
from PIL import Image
import imagepreprocessing_prototype
# Sample use which can be used to test the effectiveness of the software.
image_path = './image.png'
image = Image.open(image_path)
gray_image = numpy.asarray(image.convert('L'))
image = numpy.asarray(image)
shading_commonality = list()
# to_grayscale_jit = numba.jit("void(f4[:,:,:], f4[:])")(to_grayscale)
# blur_jit = numba.jit("void(f4[:], u8)")(blur)
# im = image_variance_art(image, 2)
#im = imagepreprocessing_prototype.image_stretcher(gray_image)
im = imagepreprocessing_prototype.image_stretcher(gray_image, 2)
# im = blur(im, 2)
im = Image.fromarray(im)
im.save('extractedFeatures.jpg')
