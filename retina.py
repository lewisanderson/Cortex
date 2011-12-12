#! /opt/local/bin/python2.7
import numpy as np, cv, cv2, time, random, math

'''
Notes: 
	http://en.wikipedia.org/wiki/Retina#Spatial_encoding
	http://grey.colorado.edu/CompCogNeuro/index.php/CCNBook/Perception#Oriented_Edge_Detectors_in_Primary_Visual_Cortex

(first success: 1.0,0.625)
(untested: 1.6,1.0)
(decent, unt: 2.56,1.6
(second atmpt: 4.096,2.56)


100x (5x at center, 1000x at edges) more photoreceptors than ganglion cells
	lets use 9pixels/ganglion, to approximate the center of a field of view
	it appears that ganglion cells overlap.
	what size of difference-of-gaussian is that? TODO
positive ganglion (on-center) cells are excitatory
negative ganlgion (off-center) cells are inhibitory
(so, we can just use one for all, that has negative outputs for negative values)

Target Image size: 500x500
	start with 500x500 ganglion cells

usage
import retina as r
input_filename = '../data/twigs.JPG'
output = r.calc_retinal_layer(input_filename, 500, 500)

'''

########## Display/Logging Functions ##########

start = time.time()
def start_timer():
	global start
	start = time.time()

def print_timer(msg):
	global start,next
	next = time.time()
	print '%.3fs' % (next - start), msg
	start = next

def print_stats(name, input):
	vals = (np.max(input),np.min(input),np.average(input),np.std(input))
	print "max:%.3f, min:%.3f, avg:%.3f, std:%.3f <-- " % vals, name

def show_dog(dog):
	''' show_dog(): shows the result of a difference of gaussian filter
	inputs: 
		dog: cvMat object representing result of a difference-of-gaussian (d-o-g or dog) filter
	returns:
		nothing
	effects:
		makes and shows a window called 'Difference-of-Gaussian'
	'''
	cv.NamedWindow('Difference-of-Gaussian')
	dogpretty = cv.CreateMat(dog.cols, dog.rows, cv.CV_32FC1)
	cv.ConvertScale(dog, dogpretty, 0.5,0.5)
	cv.ShowImage('Difference-of-Gaussian',dogpretty)

def show_image(name,image):
	''' show_image(): shows an image
	inputs: 
		name: string for name of this iamge
		image: cvMat object representing the image
	returns:
		nothing
	effects:
		makes and shows a window called name, showing image
	'''
	cv.NamedWindow(name)
	cv.ShowImage(name,image)

def show_np(name,np_arr,w,h):
	''' show_np(): shows a numpy array
	inputs: 
		name: string for name of this iamge
		np_arr: numpy array object representing the image
		w.h: ints for size of image toshow
	returns:
		nothing
	effects:
		makes and shows a window called name, showing image
	'''
	act_mat = cv.fromarray(np_arr.reshape((w,h)))
	if w*10 < 500 and h*10 < 500:
		w,h = w*10,h*10
	act_mat_big = cv.CreateMat(w, h, act_mat.type)
	cv.Resize(act_mat,act_mat_big,cv.CV_INTER_NN)
	show_image(name,act_mat_big)


def print_mat(input):
	for row in input:
		for val in input:
			print val,

def print_np(arr):
	print "[",
	if len(arr.shape) > 1:
		for row in arr:
			print "[",
			for x in row:
				print "%.4f\t"%x,
			print "]"
	else:
		for x in arr:
			print "%.4f\t"%x,
	print "]"
		

def print_array(arr):
	print np.array_str(arr,precision=4,suppress_small=True)

########## Setup Functions ##########

def gaussian(sigma=0.5, shape=None):
	"""
	Gaussian kernel numpy array with given sigma and shape.

	The shape argument defaults to ceil(6*sigma).
	"""
	sigma = max(abs(sigma), 1e-10)
	if shape is None:
		shape = max(int(6*sigma+0.5), 1)
	if not isinstance(shape, tuple):
		shape = (shape, shape)
	x = np.arange(-(shape[0]-1)/2.0, (shape[0]-1)/2.0+1e-8)
	y = np.arange(-(shape[1]-1)/2.0, (shape[1]-1)/2.0+1e-8)
	Kx = np.exp(-x**2/(2*sigma**2))
	Ky = np.exp(-y**2/(2*sigma**2))
	ans = np.outer(Kx, Ky) / (2.0*np.pi*sigma**2)
	return ans/sum(sum(ans))

def gaussian1d(sigma=2.56, size=23):
	""" gaussian1d(): make 1-D gaussian distribution
	"""
	sigma = max(abs(sigma), 1e-10)
	x = np.arange(-(size-1)/2.0, (size-1)/2.0+1e-8)
	Kx = np.exp(-x**2/(2*sigma**2))
	return Kx

def build_kernel():
	''' build_kernel(): builds the difference-of-gaussian kernel
	inputs and outputs
		none
	effects
		sets dog_2d_mat
	'''
	global dog_2d_mat#,dog_2d_mat_detailed
	wide_sigma = 2.56 # 4.096,2.56
	narrow_sigma = 1.6
	kernel_size = 23
	
	print "d-o-g kernel:(%.4f/%.4f)"%(wide_sigma,narrow_sigma)
	np_gauss_wide = gaussian(sigma=wide_sigma, shape=(kernel_size,kernel_size))
	np_gauss_narrow = gaussian(sigma=narrow_sigma, shape=(kernel_size,kernel_size))
	np_gauss = (np_gauss_wide - np_gauss_narrow)*10.0
	dog_2d_mat = cv.fromarray(np_gauss)

########## Utility Functions ##########

def load_input(filename):
	image = None
	if filename[-4:] == '.npy':
		# load numpy array into image
		image_np = np.load(filename)
		image = cv.fromarray(image_np.copy())
	else:
		# load image file into into image
		raw_image = cv.LoadImageM(filename)
		color_image = cv.CreateMat(raw_image.rows,raw_image.cols, cv.CV_32FC3)
		image = cv.CreateMat(color_image.rows, color_image.cols, cv.CV_32FC1)
		cv.ConvertScale(raw_image,color_image,1/255.0)
		cv.CvtColor(color_image,image,cv.CV_RGB2GRAY)
	return image

def crop_image(image,retina_w,retina_h):
	''' crop_image(): crop the image to the correct width
	inputs
		image: cv.Mat object
		retina_w,retina_h: ints for size of image
	outputs
		cv.Mat object of correct size
	'''
	im_x_max = image.cols - retina_w
	im_y_max = image.rows - retina_h
	im_x = int(im_x_max/2.0)
	im_y = int(im_y_max/2.0)
	if im_x_max < 0 or im_y_max < 0: raise Exception('image too small')
	roi = (im_x,im_y,retina_w,retina_h)
	image = cv.GetSubRect(image,roi)
	return image
	
########## Calculation Functions ##########

def calc_retinal_layer(input_file, retina_w, retina_h,is_random=False,detailed=False):
	''' calc_retinal_layer(): calculates the values for each ganglion cell in the retinal layer
		1) read in image
		2) resize if necessary
		3) run retinal layer:
			a) run difference-of-gaussian (Laplace seems to be similar/identical) on image
			b) extract given number of pixels
	NOTE: always call build_kernel before calling this function the first time
	inputs: 
		input_file: the filename for the input image
	returns: 
		numpy array of floats, such that each is the value of a ganglion cell
	effects:
		none
	'''
	global dog_2d_mat,dog_2d_mat_detailed
	
	if detailed: raise Exception('detailed no longer supported')
	
	# load image, convert to grayscale
	image = load_input(input_file)
	
	# do difference of gaussian filtering
	im_w,im_h = image.cols,image.rows
	dog_mat = cv.CreateMat(im_h,im_w,cv.CV_32FC1)
	dog_2d_filter = dog_2d_mat
	cv.Filter2D(image,dog_mat,dog_2d_filter)
	
	# crop image
	dog_mat_small = crop_image(dog_mat,retina_w,retina_h)
	
	# convert to numpy and cap at [-1.0,1.0]
	dog_np = np.asarray(dog_mat_small)
	dog_np = np.minimum(dog_np,np.ones(dog_np.shape))
	dog_np = np.maximum(dog_np,-np.ones(dog_np.shape))
	
#	print_stats('dog',dog_np)
#	show_image('input image',image)
#	show_image('d-o-g filtered',dog_mat)
	
	return dog_np




########## Main Script ##########

build_kernel() # must be called before calc_retinal_layer()



def t():
	''' t(): test
	'''
	start_timer()
	calc_retinal_layer('test.png',40,40)
	print_timer('done with retina calculations')

