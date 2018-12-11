import numpy as np 
import imageio
from scipy import misc

### Some of the helper functions are provided in by Prof. Maire

def load_image(filename, crop = True):
	'''
	Load an rgb image and crop the central part if required
	'''
	img = imageio.imread(filename)[:, :, :3] 
	#img = img / 255
	if crop:
		row, col = img.shape[0], img.shape[1]
		row_off, col_off = (row - 128) // 2, (col - 64) // 2
		img = img[row_off : row - row_off, col_off : col - col_off]
	return img


def pad_border(image, wx = 1, wy = 1):
	assert image.ndim == 2, 'image should be grayscale'
	sx, sy = image.shape
	img = np.zeros((sx+2*wx, sy+2*wy))
	img[wx:(sx+wx),wy:(sy+wy)] = image
	return img


def trim_border(image, wx = 1, wy = 1):
	assert image.ndim == 2, 'image should be grayscale'
	sx, sy = image.shape
	img = np.copy(image[wx:(sx-wx),wy:(sy-wy)])
	return img


def mirror_border(image, wx = 1, wy = 1):
	assert image.ndim == 2, 'image should be grayscale'
	sx, sy = image.shape
	# mirror top/bottom
	top = image[:wx:,:]
	bottom = image[(sx-wx):,:]
	img = np.concatenate( \
		(top[::-1,:], image, bottom[::-1,:]), \
		axis=0 \
	)
	# mirror left/right
	left  = img[:,:wy]
	right = img[:,(sy-wy):]
	img = np.concatenate( \
		(left[:,::-1], img, right[:,::-1]), \
		axis=1 \
	)
	return img


def conv_2d(image, filt):
	# make sure that both image and filter are 2D arrays
	assert image.ndim == 2, 'image should be grayscale'
	filt = np.atleast_2d(filt)
	# get image and filter size
	sx, sy = image.shape
	sk, sl = filt.shape
	# pad image border by filter width
	wx = (sk - 1) // 2
	wy = (sl - 1) // 2
	image = pad_border(image, wx, wy)
	# intialize convolution result
	result = np.zeros(image.shape)
	# convolve
	for x in range(wx, sx + wx):
		for y in range(wy, sy + wy):
			for k in range(sk):
				for l in range(sl):
					result[x,y] = result[x,y] + \
					image[x-wx+k, y-wy+l] * filt[sk-1-k, sl-1-l]
	# remove padding
	result = trim_border(result, wx, wy)
	return result


def sobel_gradients(image):
	# make sure that image is a 2D array
	assert image.ndim == 2
	# define filters
	fx_a = np.transpose(np.atleast_2d([1,0,-1]))
	fx_b = np.atleast_2d([1,2,1])
	fy_a = np.transpose(np.atleast_2d([1,2,1]))
	fy_b = np.atleast_2d([1,0,-1])
	# pad image by mirroring
	img = mirror_border(image, 1, 1)
	# compute gradients via separable convolution
	dx = conv_2d(conv_2d(img, fx_a), fx_b)
	dy = conv_2d(conv_2d(img, fy_a), fy_b)
	# remove padding
	dx = trim_border(dx, 1, 1)
	dy = trim_border(dy, 1, 1)
	return dx, dy
