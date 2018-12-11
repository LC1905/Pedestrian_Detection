import os
import glob
import math
from utils import *

paths = "../INRIAPerson/train_64x128_H96/pos"


def gradient(img):
	'''
	Input
		img: RGB image (height, width, 3) on [0, 1] scale
	
	Output:
		mag: gradient magnitudes (height, width)
		theta: gradient orientation (height, width)
	'''
	dxs, dys = [], []
	for channel in range(3):
		dx, dy = sobel_gradients(img[:, :, channel])
		dxs.append(dx)
		dys.append(dy)
	dxs, dys = np.array(dxs), np.array(dys)
	dx, dy = np.amax(dxs, axis = 0), np.amax(dys, axis = 0)
	mag = np.sqrt((dx * dx) + (dy * dy))
	theta = np.arctan2(dy, dx)
	return mag, theta

def bin_gradient(theta, num = 10):
	'''
	Input:
		theta (height, width)
		number of bins (+1)

	Output:
		bin for each pixel (height, width)
	'''
	bins = np.linspace(0, math.pi, num = num)
	theta[theta < 0] += math.pi
	ori = np.digitize(theta, bins) - np.ones(theta.shape)
	return ori


# def HOG(window)
	'''
	Input: 
		(128, 64) decision window
	Output: 
		(3780, ) feature descriptor 
	'''


