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


def get_boarder(r, c, r2, c2, width = 2):
	boarder = []
	for i in range(r, r2):
		for w in range(width):
			boarder.append((i, c + w))
			boarder.append((i, c2 - w))

	for j in range(c, c2):
		for w in range(width):
			boarder.append((r + w, j))
			boarder.append((r2 - w, j))
	#print("boarder length: {}".format(len(boarder)))
	return boarder


def iou(cor_a, cor_b, debug = True):

	if debug:
		print("cor_a: {}, cor_b: {}".format(cor_a, cor_b))

	area_a = (cor_a[2] - cor_a[0]) * (cor_a[3] - cor_a[1])
	area_b = (cor_b[2] - cor_b[0]) * (cor_b[3] - cor_b[1])

	#row_int = min(max(0, cor_a[2] - cor_b[0]), max(0, cor_b[2] - cor_a[0]))
	#col_int = min(max(0, cor_a[3] - cor_b[1]), max(0, cor_b[3] - cor_a[1]))

	if cor_a[2] <= cor_b[0] or cor_b[2] <= cor_a[0]:
		row_int = 0
	elif cor_b[0] < cor_a[0] and cor_a[2] < cor_b[2]:
		row_int = cor_a[2] - cor_a[0]
	elif cor_a[0] < cor_b[0] and cor_b[2] < cor_a[2]:
		row_int = cor_b[2] - cor_b[0]
	else:
		row_int = min(cor_a[2] - cor_b[0], cor_b[2] - cor_a[0])

	if cor_a[3] <= cor_b[1] or cor_b[3] <= cor_a[1]:
		col_int = 0
	elif cor_b[1] < cor_a[1] and cor_a[3] < cor_b[3]:
		col_int = cor_a[3] - cor_a[1]
	elif cor_a[1] < cor_b[1] and cor_b[3] < cor_a[3]:
		col_int = cor_b[3] - cor_b[1]
	else:
		col_int = min(cor_a[3] - cor_b[1], cor_b[3] - cor_a[1])

	intersection = row_int * col_int
	iou_score = intersection / (area_a + area_b - intersection)
	if debug and iou_score > 0:
		print("area a: {}, area b: {}".format(area_a, area_b))
		print("row intersection: {}, col intersection: {}".format(row_int, col_int))
		print("intersection: {}".format(intersection))
		print("iou score: {}".format(iou_score))
	return iou_score


def recognize(filename, model_path = "model.p", org_stride = 64, \
	level = 3, wsize = (128, 64), width = 2,
	block_size = 16, block_stride = 8, isize = (320, 240), \
	threshold = 0, top_k = None, nonmax = False):

	assert os.path.exists(model_path)
	assert os.path.exists(filename)
	model = pickle.load(open(model_path, "rb"))
	oimg = load_image(filename, crop = False)

	ratio = ((oimg.shape[1] / isize[0]) ** (1 / level), (oimg.shape[0] / isize[1]) ** (1 / level))
	#ratio = (2, 2)
	#isize = (int(oimg.shape[1] / (ratio[0] ** (level - 1))), int(oimg.shape[0] / (ratio[1] ** (level - 1))))
	print("RATIO: {}".format(ratio))

	corners = []
	scores = []
	for l in range(level + 1):
		print("-" * 70)
		scale = (ratio[0] ** (level - l), ratio[1] ** (level - l))
		stride = org_stride
		print("LEVEL: {}, SCALE: {}, STRIDE: {}".format(l, scale, stride))
		img = Image.open(filename).resize(isize)
		img = np.array(img) / 255
		print("image size: {}".format(img.shape))

		offsets = [(r, c) for r in range(0, img.shape[0] - wsize[0] + 1, stride) \
			for c in range(0, img.shape[1] - wsize[1] + 1, stride)]
		#print("offsets: {}".format(offsets))
		print("number of windows: {}".format(len(offsets)))

		'''
		### visualize all blocks
		
		boarders = []
		for r, c in offsets:
			r2, c2 = r + wsize[0], c + wsize[1]
			boarders += get_boarder(r, c, r2, c2, width = width)
		for r, c in boarders[768 * 4 : 768 * 5 + 200]:
			print("r: {}, c: {}".format(r, c))
			if 0 <= r < img.shape[0] and 0 <= c < img.shape[1]:
				img[r][c] = [1, 0, 0]
		return img
		'''
		
		r_blocks = (img.shape[0] - block_size) // block_stride + 1
		c_blocks = (img.shape[1] - block_size) // block_stride + 1 
		#print("rblocks: {}, cblocks: {}".format(r_blocks, c_blocks))

		print("extract features...")
		features = HOG(img)
		features = features.reshape((r_blocks, c_blocks, 36))
		
		r_inc = wsize[0] // block_stride - 1
		c_inc = wsize[1] // block_stride - 1

		print("predict...")

		detect_features = []
		for r, c in offsets:
			block_left, block_top = c // block_stride, r // block_stride
			block_right, block_but = block_left + c_inc, block_top + r_inc
			feature = features[block_top : block_but, block_left : block_right, :].flatten()
			detect_features.append(feature)

		detect_features = np.array(detect_features)
		print("shape of detect_features: {}".format(detect_features.shape))

		
		pred = model.decision_function(detect_features)

		print("detect...")
		if top_k is None:
			for i, p in enumerate(pred):
				if p > threshold:
					#print("AHA origin: {}, {}".format(offsets[i][0], offsets[i][1]))
					r, c = offsets[i][0] * scale[0], offsets[i][1] * scale[1] # top-left
					r2, c2 = r + wsize[0] * scale[0], c + wsize[1] * scale[1] # but-right
					corners.append((int(r), int(c), int(r2), int(c2)))
					scores.append(p)
					#print("AHA: {}".format(corners[-1]))
			print("detected corners: {}".format(corners))

		else:
			scores.append(pred)
			for r, c in offsets:
				r, c = r * scale[0], c * scale[1]
				r2, c2 = r + wsize[0] * scale[0], c + wsize[1] * scale[1]
				corners.append((int(r), int(c), int(r2), int(c2)))
		isize = (int(isize[0] * ratio[0]), int(isize[1] * ratio[1]))

	
	
	if top_k is not None:
		scores = np.concatenate(scores)
		scores_corners = sorted(zip(scores, corners), reverse = True)[:top_k]
		corners = [score[1] for score in scores_corners]
		print(scores_corners)

	else:
		scores = np.array(scores)
		scores_corners = sorted(zip(scores, corners), reverse = True)
	
	print("number of corners: {}".format(len(scores_corners)))
	if nonmax:
		print("nonmax suppression...")
		copy_scores_corners = scores_corners[:]
		scores_corners = []
		while copy_scores_corners != []:
			print("length of copy: {}".format(len(copy_scores_corners)))
			score, corner = copy_scores_corners.pop(0)
			scores_corners.append((score, corner))
			print("current max score: {} at corner: {}".format(score, corner))
			copy_scores_corners = [pair for pair in copy_scores_corners\
								if iou(corner, pair[1]) < 0.2]
	
	print("number of corners: {}".format(len(scores_corners)))
	corners = [score[1] for score in scores_corners]
	## visualize
	print("visualize...")
	for r, c, r2, c2 in corners:
		print("[({}, {}), ({}, {})]".format(r, c, r2, c2))
		oimg[r : r + width, c : c2] = [1, 0, 0]
		oimg[r2 - width : r2, c : c2] = [1, 0, 0]
		oimg[r : r2, c : c + width] = [1, 0, 0]
		oimg[r : r2, c2 - width : c2] = [1, 0, 0]
	return oimg, scores_corners

#test()	
