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
	print("-" * 32 + "STATISTICS" + "-" * 32)
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

	correctness = ((tn + tp) / predict.shape[0]) * 100
	print("correctness: {}%".format(round(correctness, 2)))
	precision, recall = (tp / (tp + fp)) * 100, (tp / (tp + fn)) * 100
	print("precision: {}%, recall: {}%".format(round(precision, 2), round(recall, 2)))
	print("tp: {}, fp: {}, tn: {}, fn: {}".format(tp, fp, tn, fn))
	print("-" * 75)
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


def recognize(filename, model_path = "model.p", stride = 64, \
	level = 3, wsize = (128, 64), width = 2,
	block_size = 16, block_stride = 8, isize = (320, 240)):

	assert os.path.exists(model_path)
	assert os.path.exists(filename)
	model = pickle.load(open(model_path, "rb"))
	#img = load_image(filename, crop = False)
	img = Image.open(filename).resize(isize)
	img = np.array(img) / 255
	print("image size: {}".format(img.shape))

	offsets = [(r, c) for r in range(0, img.shape[0] - wsize[0] + 1, stride) \
			for c in range(0, img.shape[1] - wsize[1] + 1, stride)]
	print("number of windows: {}".format(len(offsets)))

	'''
	### visualize all blocks
	
	boarders = []
	for r, c in offsets:
		boarders += get_boarder(img.shape, r, c, width = width)
	for r, c in boarders[:]:
		if 0 <= r < img.shape[0] and 0 <= c < img.shape[1]:
			img[r][c] = [1, 0, 0]
	return img
	'''
	
	r_blocks = (img.shape[0] - block_size) // block_stride + 1
	c_blocks = (img.shape[1] - block_size) // block_stride + 1 
	print("rblocks: {}, cblocks: {}".format(r_blocks, c_blocks))

	print("extract features...")
	features = HOG(img)
	features = features.reshape((r_blocks, c_blocks, 36))

	### Previous computation of feature vectors
	#detect_features = [HOG(img[cood[0] : cood[0] + wsize[0], cood[1] : cood[1] + wsize[1]]) \
			#for cood in offsets]
	
	r_inc = wsize[0] // block_stride - 1
	c_inc = wsize[1] // block_stride - 1

	print("predict...")
	detect_features = []
	for r, c in offsets:
		block_left, block_top = c // block_stride, r // block_stride
		block_right, block_but = block_left + c_inc, block_top + r_inc
		#print("block left: {}, block top: {}".format(block_left, block_top))
		#print("block right: {}, block buttom: {}".format(block_right, block_but))
		feature = features[block_top : block_but, block_left : block_right, :].flatten()
		detect_features.append(feature)

	detect_features = np.array(detect_features)
	print("shape of detect_features: {}".format(detect_features.shape))

	detect_features = np.array(detect_features)
	pred = model.predict(detect_features)

	print("detect...")
	boarders = []
	corners = []
	for i, p in enumerate(pred):
		if p == 1:
			r, c = offsets[i][0], offsets[i][1] # top-left
			r2, c2 = r + wsize[0], c + wsize[1] # but-right
			corners.append((r, c, r2, c2))
			boarders += get_boarder(img.shape, r, c, width = width)

	print("detected corners: {}".format(corners))
	for r, c in boarders:
		img[r][c] = [1, 0, 0]
	return img

#test()	
