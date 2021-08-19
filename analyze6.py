# requires: pip install imagecodecs
# requires: pip install opencv-python

import imageio
from multiprocessing import Pool
import numpy as np
import re
import sys
import warnings

import imageops
import keyence

class Image:
	def __init__(self, filename):
		self.fl_filename = filename
		self.bf_filename = filename.replace('CH1', 'CH4')

		match = re.search(r'([a-zA-Z0-9]+)_XY([0-9][0-9])_', filename)
		if not match:
			raise ValueError('Filename %s missing needed xy information' % filename)

		self.plate = match.group(1)
		self.xy = int(match.group(2))

		self.well = keyence.xy_to_well(self.xy)
		self.column = self.well[0]
		self.row = self.well[1:]

		self.bf_img = None
		self.fl_img = None
		self.mask = None
		self.normalized_value = None
		self.value = None

	def get_bf_img(self):
		if self.bf_img is None:
			with warnings.catch_warnings():
				warnings.simplefilter("ignore", UserWarning)
				self.bf_img = imageio.imread(self.bf_filename)
		return self.bf_img

	def get_fl_img(self):
		if self.fl_img is None:
			with warnings.catch_warnings():
				warnings.simplefilter("ignore", UserWarning)
				self.fl_img = imageio.imread(self.fl_filename)[:,:,1]
		return self.fl_img

	def get_mask(self, silent=True, verbose=False):
		if self.mask is None:
			self.mask = imageops.get_fish_mask(self.get_bf_img(), silent, verbose)
		return self.mask

	def get_raw_value(self, silent=True, verbose=False):
		if self.value is None:
			fl_img_masked = imageops.apply_mask(self.get_fl_img(), self.get_mask(silent, verbose))
			total = fl_img_masked.sum(dtype=np.uint64, where=(fl_img_masked>5_000))
			self.value = total if total > 0 else np.nan
		return self.value

	def normalize(self, control_values):
		try:
			val = float(self.get_raw_value() * 100 // control_values[self.plate])
			self.normalized_value = val if val < 150 else np.nan # discard results >=150% of ctrl
		except ZeroDivisionError:
			print('ERROR: Plate', self.plate, 'column', self.column, 'with value',
				self.get_raw_value(), 'has control value', control_values[self.plate])
			self.normalized_value = np.nan
		return self

def quantify(filenames):
	images = [Image(filename) for filename in filenames]
	_ = Pool(8).map(Image.get_raw_value, images)
	control_values = _calculate_control_values(images)
	return [image.normalize(control_values) for image in images]

def _calculate_control_values(images):
	ctrl_imgs = [img for img in images if img.column == 'B']
	ctrl_vals = {}

	for plate in np.unique([img.plate for img in ctrl_imgs]):
		ctrl_results = np.array([img.get_raw_value() for img in ctrl_imgs if img.plate == plate])
		while True:
			ctrl_vals[plate] = float(np.nanmean(ctrl_results))
			upper = ctrl_vals[plate] * 1.5
			lower = ctrl_vals[plate] * 0.5

			valid_indices = (ctrl_results <= upper) & (ctrl_results >= lower)
			if np.all(valid_indices):
				break
			ctrl_results = ctrl_results[valid_indices]

	return ctrl_vals

if __name__ == '__main__':
	if len(sys.argv) > 1:
		images = quantify(sys.argv[1:])
		for col in keyence.COLUMNS:
			relevant_values = [img.normalized_value for img in images if img.column == col]
			print(col, np.nanmean(relevant_values), relevant_values)
	else:
		raise TypeError('Invoke with an argument, i.e. the name of a file or files to analyze.')
