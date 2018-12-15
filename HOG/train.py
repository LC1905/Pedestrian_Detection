import os
import glob
import numpy as np
import pickle
import test
import random
from features import *
from utils import *
from sklearn.svm import LinearSVC

PATH = "../INRIAPerson/train_64x128_H96"

def feature_extraction(split, rload = True, save = True):
	'''
	Input
		split: pos or neg
	'''
	feature_path = "train_" + split + ".npy"
	print("feature path: {}".format(feature_path))
	if os.path.exists(feature_path) and rload:
		print("reload features...")
		X = np.load(feature_path)
	else:
		X = []
		filenames = glob.glob(os.path.join(PATH, split) + "/*.png")
		for i, filename in enumerate(filenames):
			print("image: {}".format(i))
			if split == "pos":
				img = load_image(filename, crop = True)
				feature = HOG(img)
				X.append(feature)
			else:
				img = load_image(filename, crop = False)
				for k in range(10):
					si = random.randint(0, img.shape[0] - 128)
					sj = random.randint(0, img.shape[1] - 64)
					window = img[si : si + 128, sj : sj + 64]
					feature = HOG(window)
					X.append(feature)

		X = np.array(X)
		print("shape of features: {}".format(X.shape))
		if save:
			np.save(feature_path, X)

	if split == "pos":
		y = np.ones(X.shape[0])
	else:
		y = np.zeros(X.shape[0])

	return X, y


def train_model(model_path = "model.p", rload = True, save = True):
	if os.path.exists(model_path) and rload:
		model = pickle.load(open(model_path, "rb"))
		return model

	pos_X, pos_y = feature_extraction("pos")
	neg_X, neg_y = feature_extraction("neg")
	X = np.concatenate([pos_X, neg_X])
	y = np.concatenate([pos_y, neg_y])
	print("shape of X: {}, shape of y: {}".format(X.shape, y.shape))
	model = LinearSVC(C = 0.01)
	print("train model...")
	model.fit(X, y)
	if save:
		pickle.dump(model, open(model_path, "wb"))


def first_test_train():
	model = train_model()
	pos_X, pos_y = feature_extraction("pos")
	neg_X, neg_y = feature_extraction("neg")
	X = np.concatenate([pos_X, neg_X])
	y = np.concatenate([pos_y, neg_y])
	s = model.score(X, y)
	print("accuracy: {}".format(s))

'''
def first_test_test():
	model = train_model()
	pos_X, pos_y = test.test_feature_extraction("pos")
	neg_X, neg_y = test.test_feature_extraction("neg")
	X = np.concatenate([pos_X, neg_X])
	y = np.concatenate([pos_y, neg_y])
	s = model.score(X, y)
	print("accuracy: {}".format(s))

first_test_test()
'''
