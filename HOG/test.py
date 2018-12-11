import os
import glob
import numpy as np
from features import *
from utils import *

PATH = "../INRIAPerson/test_64x128_H96"

def test_feature_extraction(split, rload = True, save = True):
	'''
	Input
		split: pos or neg
	'''
	feature_path = "test_" + split + ".npy"
	if os.path.exists(feature_path) and rload:
		print("reload features...")
		X = np.load(feature_path)
	else:
		X = []
		filenames = glob.glob(os.path.join(PATH, split) + "/*.png")
		for i, filename in enumerate(filenames):
			print("image: {}".format(i))
			img = load_image(filename, crop = True)
			feature = HOG(img)
			X.append(feature)
		X = np.array(X)
		print("shape of features: {}".format(X.shape))
		if save:
			np.save(feature_path, X)

	if split == "pos":
		y = np.ones(X.shape[0])
	else:
		y = np.zeros(X.shape[1])

	return X, y


