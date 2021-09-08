# requires: pip install imagecodecs
# requires: pip install opencv-python

import argparse
import csv
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
import pandas as pd
import re
import seaborn as sns
import sys
import warnings

import imageops
import keyence

class Image:
	def __init__(self, filename, group, debug=0):
		self.fl_filename = filename
		self.bf_filename = filename.replace('CH1', 'CH4')

		match = re.search(r'([a-zA-Z0-9]+)_XY([0-9][0-9])_', filename)
		if not match:
			raise UserError('Filename %s missing needed xy information' % filename)

		self.plate = match.group(1)
		self.xy = int(match.group(2))

		self.group = group
		self.debug = debug

		self.bf_img = None
		self.fl_img = None
		self.mask = None
		self.normalized_value = None
		self.value = None

	def get_bf_img(self):
		if self.bf_img is None:
			with warnings.catch_warnings():
				warnings.simplefilter("ignore", UserWarning)
				self.bf_img = imageops.read(self.bf_filename, np.uint16)
		return self.bf_img

	def get_fl_img(self):
		if self.fl_img is None:
			with warnings.catch_warnings():
				warnings.simplefilter("ignore", UserWarning)
				self.fl_img = imageops.read(self.fl_filename, np.uint16, 1)
		return self.fl_img

	def get_mask(self):
		if self.mask is None:
			self.mask = imageops.get_fish_mask(
				self.get_bf_img(), self.get_fl_img(), True, self.debug < 1, self.debug >= 2,
				'{}_XY{:02d}'.format(self.plate, self.xy))
		return self.mask

	def get_raw_value(self, threshold=0.05):
		if self.value is None:
			fl_img_masked = imageops.apply_mask(self.get_fl_img(), self.get_mask())
			fl_max_pixel = fl_img_masked.max()
			total = fl_img_masked.sum(
				dtype=np.uint64, where=(fl_img_masked > fl_max_pixel*threshold))
			self.value = total if total > 0 else np.nan
		return self.value

	def normalize(self, control_values, cap):
		try:
			val = float(self.get_raw_value() * 100 // control_values[self.plate])
			if cap > 0:
				self.normalized_value = val if val < cap else np.nan
			else:
				self.normalized_value = val
		except ZeroDivisionError:
			print('ERROR: Plate', self.plate, 'group', self.group, 'with value',
				self.get_raw_value(), 'has control value', control_values[self.plate])
			self.normalized_value = np.nan
		return self

class UserError(ValueError):
	pass

def chart(results, chartfile):
	sns.set_theme(style='whitegrid')

	data = pd.DataFrame({
		'brightness': [value for values in results.values() for value in values],
		'group': [key for key, values in results.items() for _ in values],
	})

	sns.swarmplot(x='group', y='brightness', data=data)
	sns.boxplot(x='group', y='brightness', data=data, showbox=False, showcaps=False,
		showfliers=False, whiskerprops={'visible': False})
	plt.xticks(rotation=45)
	plt.tight_layout()
	plt.savefig(chartfile)

def get_schematic(platefile, target_count, plate_ignore):
	if not platefile:
		return keyence.LAYOUT_DEFAULT

	if '' not in plate_ignore:
		plate_ignore.append('')

	with open(platefile, encoding='utf8', newline='') as f:
		schematic = [[well for well in row if well not in plate_ignore] for row in csv.reader(f)]

	count = sum([len(row) for row in schematic])
	if count != target_count:
		del schematic[0]
		for row in schematic:
			del row[0]
		count = sum([len(row) for row in schematic])
		if count != target_count:
			raise UserError('Schematic does not have same number of cells as images provided')

	return [well for row in schematic for well in row]

def main(imagefiles, cap=150, chartfile=None, debug=0, group_regex='.*', platefile=None,
		plate_control=['B'], plate_ignore=[], silent=False):
	results = {}

	schematic = get_schematic(platefile, len(imagefiles), plate_ignore)
	groups = list(dict.fromkeys(schematic))
	images = quantify(imagefiles, plate_control, cap=cap, debug=debug, group_regex=group_regex,
		schematic=schematic)

	pattern = re.compile(group_regex)
	for group in groups:
		if group in plate_control or pattern.search(group):
			relevant_values = [img.normalized_value for img in images if img.group == group]
			results[group] = relevant_values
			if not silent:
				with warnings.catch_warnings():
					warnings.simplefilter("ignore", RuntimeWarning)
					print(group, np.nanmedian(relevant_values), relevant_values)

	if chartfile:
		chart(results, chartfile)

	return results

def quantify(imagefiles, plate_control=['B'], cap=150, debug=0, group_regex='.*', schematic=None):
	pattern = re.compile(group_regex)
	images = [Image(filename, group, debug) for filename, group in zip(imagefiles, schematic)
		if group in plate_control or pattern.search(group)]
	control_values = _calculate_control_values(images, plate_control)
	return [image.normalize(control_values, cap) for image in images]

def _calculate_control_values(images, plate_control):
	ctrl_imgs = [img for img in images if img.group in plate_control]
	ctrl_vals = {}

	for plate in np.unique([img.plate for img in ctrl_imgs]):
		ctrl_results = np.array([img.get_raw_value() for img in ctrl_imgs if img.plate == plate])
		ctrl_vals[plate] = float(np.nanmedian(ctrl_results))

	if not ctrl_vals:
		raise UserError(
			'No control wells found. Please supply a --plate-control, or modify the given value.')

	return ctrl_vals

#
# main
#

if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description=('Analyzer for images of whole zebrafish with stained neuromasts, for the '
			'purposes of measuring hair cell damage.'))

	parser.add_argument('imagefiles',
		nargs='+',
		help='The absolute or relative filenames where the relevant images can be found.')
	parser.add_argument('-ch', '--chartfile',
		help='If supplied, the resulting numbers will be charted at the given filename.')

	parser.add_argument('-p', '--platefile',
		help='CSV file containing a schematic of the plate from which the given images were '
			'taken. Row and column headers are optional. The cell values are essentially just '
			'arbitrary labels: results will be grouped and charted according to the supplied '
			'values.')
	parser.add_argument('-pc', '--plate-control',
		default=['B'],
		nargs='*',
		help='Labels to treat as the control condition in the plate schematic. These wells are '
			'used to normalize all values in the plate for more interpretable results. Any number '
			'of values may be passed.')
	parser.add_argument('-pi', '--plate-ignore',
		default=[],
		nargs='*',
		help='Labels to ignore (treat as null/empty) in the plate schematic. Empty cells will '
			'automatically be ignored, but any other null values (e.g. "[empty]") must be '
			'specified here. Any number of values may be passed.')

	parser.add_argument('-g', '--group-regex',
		default='.*',
		help=('Pattern to be used to match group names that should be included in the results. '
			'Matched groups will be included, groups that don\'t match will be ignored. Control '
			'wells will always be included regardless of whether they match.'))

	parser.add_argument('-c', '--cap',
		default=150,
		type=int,
		help=('Exclude well values larger than the given integer, expressed as a percentage of '
			'the median control value. Defaults to 150 (i.e. values larger than 150%% of control '
			'will be excluded.'))
	parser.add_argument('-nc', '--no-cap',
		action='store_const',
		const=-1,
		dest='cap',
		help=('If present, well values will not be excluded just by virtue of being too large.'))

	parser.add_argument('-d', '--debug',
		action='count',
		default=0,
		help=('Indicates intermediate processing images should be output for troubleshooting '
			'purposes. Including this argument once will yield one intermediate image per input '
			'file, twice will yield several intermediate images per input file.'))
	parser.add_argument('-s', '--silent',
		action='store_true',
		help=('If present, printed output will be suppressed. More convenient for programmatic '
			'execution.'))

	args = parser.parse_args(sys.argv[1:])
	args_dict = vars(args)
	try:
		main(**args_dict)
	except UserError as ue:
		print('Error:', ue)
		sys.exit(1)
