# requires: pip install imagecodecs
# requires: pip install opencv-python

import imageio
import json
from multiprocessing import Pool
import numpy as np
import pandas as pd
import re
import sys

import imageops
import keyence

def quantify(filenames):
	file_results = do_quantify4(filenames)

	results = pd.DataFrame([_get_info(filename, value) for filename, value in file_results.items()], columns=["plate", "column", "value"])

	control_values = _calculate_control_values(results)

	normalized_results = results.apply(lambda x: _normalize(x, control_values), axis=1)

	for col in keyence.COLUMNS:
		relevant = normalized_results[normalized_results['column'] == col]['value']
		print(col, relevant.mean(), relevant.values)

def quantify_4_plus(filename):
	img_array = imageio.imread(filename)[:,:,1]
	mask = imageops.get_fish_mask(filename.replace('CH1', 'CH4'), False, False)
	img_array = imageops.apply_mask(img_array, mask)
	total = img_array.sum(dtype=np.uint64, where=(img_array>5_000))
	return total if total > 0 else np.nan

def do_quantify4(filenames):
	return {filename: quantify_4_plus(filename) for filename in filenames}

def do_quantify5(filenames):
	pool = Pool(8)
	return {filename: result for filename, result in zip(filenames, pool.map(quantify_4_plus, filenames))}

def _calculate_control_values(results):
	control_values = {}

	for plate in np.unique(results['plate']):
		plate_control_results = results[(results['plate'] == plate) & (results['column'] == 'B')]

		while True:
			control_values[plate] = float(plate_control_results.mean(numeric_only=True))
			thresh_upper = control_values[plate] * 1.5
			thresh_lower = control_values[plate] * 0.5

			if not plate_control_results[(plate_control_results['value'] > thresh_upper) | (plate_control_results['value'] < thresh_lower)].any(axis=None):
				break

			plate_control_results = plate_control_results[(plate_control_results['value'] <= thresh_upper) & (plate_control_results['value'] >= thresh_lower)]

	return control_values

def _get_info(filename, value):
	match = re.search(r'([a-zA-Z0-9]+)_XY([0-9][0-9])_', filename)
	if not match:
		raise ValueError('Filename %s missing needed xy information' % filename)

	plate = match.group(1)
	column = keyence.xy_to_well(int(match.group(2)))[0]

	return (plate, column, value)

def _normalize(entry, control_values):
	try:
		new_value = float(entry['value'] * 100 // control_values[entry['plate']])
		new_value = new_value if new_value < 150 else np.nan # discard results greater than 150% of control value
	except ZeroDivisionError:
		print('ERROR: Plate', entry['plate'], 'column', entry['column'], 'with value', entry['value'], 'has control value', control_values[entry['plate']])
		new_value = np.nan

	return pd.Series([entry['plate'], entry['column'], new_value], index=entry.index)

if __name__ == '__main__':
	if len(sys.argv) > 1:
		quantify(sys.argv[1:])
	else:
		raise TypeError('Invoke with an argument, i.e. the name of a file or files to analyze.')
