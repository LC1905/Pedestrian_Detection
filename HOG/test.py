import os
import glob
import numpy as np
from features import *
from utils import *
import random
import copy
import pickle
from PIL import Image

PATH = "../INRIAPerson/test_64x128_H96"

def test_feature_extraction(split, rload = True, save = True):
	'''
	Input
		split: pos or neg
	'''
	feature_path = "test_" + split + ".npy"
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

def meta_statistics(predict, label):
	print("getting statistics...")
	assert predict.shape == label.shape
	tp, fp, tn, fn = 0, 0, 0, 0
	true_pos, false_pos, true_neg, false_neg = [], [], [], []
	for i, p in enumerate(predict):
		if p == 0:
			if label[i] == 0:
				true_neg.append(i)
				tn += 1
			else:
				false_neg.append(i)
				fn += 1
		else:
			if label[i] == 0:
				false_pos.append(i)
				fp += 1
			else:
				true_pos.append(i)
				tp += 1

	print("correctness: {}".format((tn + tp) / predict.shape[0]))
	print("tp: {}, fp: {}, tn: {}, fn: {}".format(tp, fp, tn, fn))
	return true_pos, false_pos, true_neg, false_neg


def test(model_path = "model.p"):
	print("testing...")
	assert os.path.exists(model_path)
	model = pickle.load(open(model_path, "rb"))
	pos_X, pos_y = test_feature_extraction("pos")
	neg_X, neg_y = test_feature_extraction("neg")

	X = np.concatenate([pos_X, neg_X])
	y = np.concatenate([pos_y, neg_y])
	pred = model.predict(X)
	print("shape of pred: {}".format(pred.shape))
	np.save("predict.npy", pred)
	true_pos, false_pos, true_neg, false_neg = meta_statistics(pred, y)
	return X, true_pos, false_pos, true_neg, false_neg


def get_boarder(isize, r, c, wsize = (128, 64), width = 2):
	boarder = []
	for i in range(wsize[0]):
		for w in range(width):
			boarder.append((r + i, c + w))
			boarder.append((r + i, c + wsize[1] - w))

	for j in range(wsize[1]):
		for w in range(width):
			boarder.append((r + w, c + j))
			boarder.append((r + wsize[0] - w, c + j))
	return boarder


def recognize(filename, model_path = "model.p", stride = 64, isize = (320, 240), wsize = (128, 64), width = 2):
	assert os.path.exists(model_path)
	assert os.path.exists(filename)
	model = pickle.load(open(model_path, "rb"))
	img = Image.open(filename).resize(isize)
	img = np.array(img) / 255
	print("image size: {}".format(img.shape))

	offsets = [(r, c) for r in range(0, img.shape[0] - wsize[0], stride) \
			for c in range(0, img.shape[1] - wsize[1], stride)]

	print("extract features...")
	# TODO: more efficient feature extraction
	features = [HOG(img[cood[0] : cood[0] + wsize[0], cood[1] : cood[1] + wsize[1]]) \
			for cood in offsets]

	print("predict...")
	pred = model.predict(features)

	print("detect...")
	boarders = []
	for i, p in enumerate(pred):
		if p == 1:
			r, c = offsets[i][0], offsets[i][1]
			print("AHA: {}, {}".format(r, c))
			boarders += get_boarder(img.shape, r, c, width = width)
	for r, c in boarders:
		img[r][c] = [1, 0, 0]
	return img

#test()	
