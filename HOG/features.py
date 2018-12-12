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

def bin_gradient(theta, num = 9):
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
	return ori.astype(int)

def l2_normalization(v):
	eps = 10 ** (-20) 
	norm = np.sum(v * v)
	v = v / np.sqrt(norm + eps ** 2)
	return v


def contrast_normalization(block):
	'''
	use L2-Hys norm
	'''
	eps = 10 ** (-20) 
	v = l2_normalization(block)
	v[v > 0.2] = 0.2 	# clip
	return l2_normalization(v)


def HOG(window, cell_size = 8, block_size = 16, stride = 8, num_bin = 9):
	'''
	Input: 
		(128, 64) decision window
	Output: 
		(3780, ) feature descriptor 
	'''

	mag, theta = gradient(window)
	ori = bin_gradient(theta)
	features = []
	cell_rows = mag.shape[0] // cell_size
	cell_cols = mag.shape[1] // cell_size
	#print("cell_rows: {}, cell_cols: {}".format(cell_rows, cell_cols))

	# vote, each 8 * 8 cell is associated with a (9,) feature vector
	cell_f = np.zeros((cell_rows, cell_cols, num_bin))
	for i in range(cell_rows):
		for j in range(cell_cols):
			for oi in range(cell_size):
				for oj in range(cell_size):
					x = cell_size * i + oi
					y = cell_size * j + oj
					cell_f[i][j][ori[x][y]] += mag[x][y]

	# normalize, this part might need to be generalized
	for bi in range(0, mag.shape[0] - block_size + 1, stride):
		for bj in range(0, mag.shape[1] - block_size + 1, stride):
			#print("bi: {}, bj: {}".format(bi // stride, bj // stride))
			ci, cj = bi // cell_size, bj // cell_size
			feature = contrast_normalization(np.concatenate\
				([cell_f[ci][cj], cell_f[ci][cj + 1],\
				cell_f[ci + 1][cj], cell_f[ci + 1][cj + 1]]))
			features.append(feature)
	return np.concatenate(features)


# TODO: Visualize HO
def visualize_HOG(window):
	mag, theta = gradient(window)
	ori = bin_gradient(theta)
	features = []
	cell_rows = mag.shape[0] // cell_size
	cell_cols = mag.shape[1] // cell_size