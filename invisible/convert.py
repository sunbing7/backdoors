# Import the necessary libraries
from PIL import Image
from numpy import asarray
 
 
# load the image and convert into
# numpy array
img = Image.open('sunjun.jpg')
 
# asarray() class is used to convert
# PIL images into NumPy arrays
numpydata = asarray(img)
 
# <class 'numpy.ndarray'>
print(type(numpydata))
 
#  shape
print(numpydata.shape)


print(str(numpydata.tolist()))