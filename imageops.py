import cv2 as cv
import imageio
import numpy as np
import os
import sys
import time
import warnings

def apply_mask(img, mask):# mask should have black background, white foreground
	img_type = _get_bit_depth(img)
	return np.minimum(img, np.where(mask == 255, img_type[1], 0).astype(img_type[0]))

def binarize(img, threshold):
	if threshold >= 0:
		return np.where(img < threshold, 255, 0).astype(np.uint8)
	else:
		return img

def close(img, size=1, iterations=1):
	if size > 0 and iterations > 0:
		return cv.morphologyEx(img, cv.MORPH_CLOSE, _get_kernel(size), iterations=iterations)
	else:
		return img

def dilate(img, size=1, iterations=1):
	if size > 0 and iterations > 0:
		return cv.dilate(img, _get_kernel(size), iterations=iterations)
	else:
		return img

def erode(img, size=1, iterations=1):
	if size > 0 and iterations > 0:
		return cv.erode(img, _get_kernel(size), iterations=iterations)
	else:
		return img

def get_aspect_mask(img, target_ratio=5, error_bound=0.5, verbose=False):
	contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
	upper, lower = target_ratio * (1 + error_bound), target_ratio / (1 + error_bound)
	contours = [contour for contour in contours if _aspect_is_between(contour, upper, lower)]
	return cv.drawContours(
		np.ones(img.shape, dtype=np.uint8)*255, contours, -1, (0,255,0), cv.FILLED)

def get_contours_by_area(img, threshold=-1, lower=0, upper=2**32):
	binarized_img = binarize(img, threshold)
	contours, _ = cv.findContours(binarized_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
	areas = np.array([cv.contourArea(contour) for contour in contours])
	with warnings.catch_warnings():
		warnings.simplefilter("ignore", np.VisibleDeprecationWarning)
		return np.multiply(contours, np.minimum(areas > lower, areas < upper))# ugly: consider fix

def get_fish_mask(filename, verbose=False, silent=True):
	with warnings.catch_warnings():
		warnings.simplefilter("ignore", UserWarning)
		img = imageio.imread(filename)
	show(img, verbose)
	steps = (
		rescale_brightness,
		lambda img_i: binarize(img_i, threshold=2**14),
		lambda img_i: apply_mask(
			img_i, get_size_mask(img, erosions=10, threshold=2**12, lower=2**15, verbose=verbose)),
		lambda img_i: close(img_i, size=6, iterations=16),
		lambda img_i: dilate(img_i, size=5, iterations=6),
		lambda img_i: get_size_mask(
			img_i, erosions=4, threshold=-1, lower=2**15, upper=2**19, verbose=verbose),
		invert,
		lambda img_i: get_aspect_mask(img_i, target_ratio=5, error_bound=0.5, verbose=verbose),
		invert,
	)
	mask = _get_mask(img, steps, verbose)
	if not verbose and not silent:
		show(img)
		show(apply_mask(img, mask))
	return mask

def get_size_mask(img, erosions=0, threshold=2**7, lower=0, upper=2**32, verbose=False):
	contours = get_contours_by_area(img, threshold, lower, upper)
	steps = (
		lambda img_i: cv.drawContours(
			np.ones(img_i.shape, dtype=np.uint8)*255, contours, -1, (0,255,0), cv.FILLED),
		lambda img_i: erode(img_i, size=4, iterations=erosions),
	)
	return _get_mask(img, steps, verbose)

def invert(img):
	return np.subtract(_get_bit_depth(img)[1], img)

def rescale_brightness(img):
	img_type = _get_bit_depth(img)
	return ((img - img.min()) * (img_type[1] / img.max())).astype(img_type[0])

def resize(img, factor):
	return cv.resize(img, None, fx=factor, fy=factor)

def show(img, verbose=True):
	if verbose:
		unique_str = str(int(time.time() * 1000) % 1_620_000_000_000)
		filename = '/mnt/c/Users/ethan/Pictures/zebrafish/' + unique_str + '.png'
		imageio.imwrite(filename, resize(img, 0.5))

def _aspect_is_between(contour, upper, lower):
	_, (minor, major), _ = cv.fitEllipse(contour)
	aspect_ratio = major / minor
	return (aspect_ratio < upper) and (aspect_ratio > lower)

def _get_bit_depth(img):
	types = [(itype, np.iinfo(itype).max) for itype in [np.uint8, np.uint16, np.int32]]
	return types[np.digitize(img.max(), [itype[1] for itype in types], right=True)]

def _get_kernel(size):
	return cv.getStructuringElement(cv.MORPH_ELLIPSE, (size*2 + 1, size*2 + 1), (size, size))

def _get_mask(img, steps, verbose=False):
	img_i = img
	for step in steps:
		img_i = step(img_i)
		show(img_i, verbose)
	show(apply_mask(img, img_i), verbose)
	return img_i

def _test():
	assert _get_bit_depth(np.array([1, 2, 3, 4, 5])) == (np.uint8, 255)
	assert _get_bit_depth(np.array([1, 2, 3, 4, 255])) == (np.uint8, 255)
	assert _get_bit_depth(np.array([1, 2, 3, 4, 256])) == (np.uint16, 65_535)
	assert _get_bit_depth(np.array([1, 2, 3, 4, 65_536])) == (np.int32, 2_147_483_647)

if __name__ == '__main__':
	if len(sys.argv) > 1:
		for filename in sys.argv[1:]:
			get_fish_mask(filename, True)
	else:
		raise TypeError('Invoke with an argument, i.e. the name of a file or files to process.')
