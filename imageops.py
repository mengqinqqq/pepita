import argparse
import cv2 as cv
from skimage import feature
import imageio
import numpy as np
import os
import sys
import time
import warnings

LOG_DIR = '/mnt/c/Users/ethan/Pictures/zebrafish/'

def apply_mask(img, mask):# mask should have black background, white foreground
	img_type = _get_bit_depth(img)
	return np.minimum(img, np.where(mask == 255, img_type[1], 0).astype(img_type[0]))

def binarize(img, threshold):
	if threshold >= 0:
		return np.where(img < threshold, 255, 0).astype(np.uint8)
	else:
		return img

def circle_local_maxima(img, count=50, min_pct=0.05, radius=8):
	coordinates = feature.peak_local_max(img, min_distance=radius, num_peaks=count,
		threshold_rel=min_pct)
	maxima = np.zeros_like(img)
	for coordinate in coordinates:
		maxima[coordinate[0]][coordinate[1]] = 255
	return dilate(maxima, size=radius)

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

def get_fish_mask(bf_img, fl_img, particles=True, silent=True, verbose=False, v_file_prefix=''):
	show(bf_img, verbose, v_file_prefix=v_file_prefix)
	if particles:
		show(fl_img, verbose, v_file_prefix=v_file_prefix)
	steps = (
		rescale_brightness,
		lambda img_i: binarize(img_i, threshold=2**14),
		lambda img_i: apply_mask(
			img_i, get_size_mask(bf_img, erosions=10, threshold=2**12, lower=2**15, verbose=verbose,
				v_file_prefix=v_file_prefix)),
		lambda img_i: close(img_i, size=6, iterations=16),
		lambda img_i: dilate(img_i, size=5, iterations=6),
		lambda img_i: get_size_mask(
			img_i, erosions=4, threshold=-1, lower=2**15, upper=2**19, verbose=verbose,
			v_file_prefix=v_file_prefix),
		invert,
	)
	if particles:
		steps = (
			*steps,
			lambda img_i: apply_mask(fl_img, img_i),
			circle_local_maxima,
		)
	mask = _get_mask(bf_img, steps, verbose, v_file_prefix=v_file_prefix)
	if not verbose and not silent:
		show(bf_img, v_file_prefix=v_file_prefix)
		if particles:
			show(fl_img, v_file_prefix=v_file_prefix)
		show(apply_mask(bf_img if not particles else fl_img, mask), v_file_prefix=v_file_prefix)
	return mask

def get_size_mask(img, erosions=0, threshold=2**7, lower=0, upper=2**32, verbose=False,
		v_file_prefix=''):
	contours = get_contours_by_area(img, threshold, lower, upper)
	steps = (
		lambda img_i: cv.drawContours(
			np.ones(img_i.shape, dtype=np.uint8)*255, contours, -1, (0,255,0), cv.FILLED),
		lambda img_i: erode(img_i, size=4, iterations=erosions),
	)
	return _get_mask(img, steps, verbose, v_file_prefix=v_file_prefix)

def invert(img):
	return np.subtract(_get_bit_depth(img)[1], img)

def read(filename, target_bit_depth, channel=0):
	img = imageio.imread(filename)
	if channel > 0:
		img = img[:,:,channel]

	bit_depth = _get_bit_depth(img)
	if bit_depth[0] != target_bit_depth:
		img = (img * (np.iinfo(target_bit_depth).max / bit_depth[1])).astype(target_bit_depth)

	return img

def rescale_brightness(img):
	img_type = _get_bit_depth(img)
	return ((img - img.min()) * (img_type[1] / img.max())).astype(img_type[0])

def resize(img, factor):
	return cv.resize(img, None, fx=factor, fy=factor)

def show(img, verbose=True, v_file_prefix=''):
	if verbose:
		unique_str = str(int(time.time() * 1000) % 1_620_000_000_000)
		filename = LOG_DIR + v_file_prefix + '_' + unique_str + '.png'
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

def _get_mask(img, steps, verbose=False, v_file_prefix=''):
	img_i = img
	for step in steps:
		img_i = step(img_i)
		show(img_i, verbose, v_file_prefix=v_file_prefix)
	show(apply_mask(img, img_i), verbose, v_file_prefix=v_file_prefix)
	return img_i

def _test():
	assert _get_bit_depth(np.array([1, 2, 3, 4, 5])) == (np.uint8, 255)
	assert _get_bit_depth(np.array([1, 2, 3, 4, 255])) == (np.uint8, 255)
	assert _get_bit_depth(np.array([1, 2, 3, 4, 256])) == (np.uint16, 65_535)
	assert _get_bit_depth(np.array([1, 2, 3, 4, 65_536])) == (np.int32, 2_147_483_647)

#
# main
#

def main(imagefiles, debug=1, logfile_prefix='imageops', particles=True):
	for bf_filename in imagefiles:
		fl_filename = bf_filename.replace('CH4', 'CH1')
		with warnings.catch_warnings():
			warnings.simplefilter("ignore", UserWarning)
			bf_img = read(bf_filename, np.uint16)
			fl_img = None if not particles else read(fl_filename, np.uint16, 1)
		get_fish_mask(bf_img, fl_img, particles=particles, silent=debug<1, verbose=debug>1,
			v_file_prefix=logfile_prefix)

if __name__ == '__main__':
	_test()

	parser = argparse.ArgumentParser(
		description=('Utility for operating on images of whole zebrafish with stained neuromasts, '
			'for the purposes of measuring hair cell damage.'))

	parser.add_argument('imagefiles',
		nargs='+',
		help='The absolute or relative filenames where the relevant images can be found.')
	parser.add_argument('-p', '--particles',
		action='store_true',
		help=('If present, the resulting mask will obscure everything except the bright particles '
			'on the fish in the given images. Otherwise the whole fish will be shown.'))
	parser.add_argument('-d', '--debug',
		action='count',
		default=1,
		help=('Indicates intermediate processing images should be output for troubleshooting '
			'purposes. Including this argument once will yield one intermediate image per input '
			'file, twice will yield several intermediate images per input file.'))

	args = parser.parse_args(sys.argv[1:])
	args_dict = vars(args)
	main(**args_dict)
