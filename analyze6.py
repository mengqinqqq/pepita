# requires: pip install imagecodecs
# requires: pip install opencv-python

import imageio
import json
from multiprocessing import Pool
import numpy as np
import pandas as pd
import re
import sys
import warnings

import imageops
import keyence

def main(filenames):
	normalized_results = quantify(filenames)

	for col in keyence.COLUMNS:
		relevant = normalized_results[normalized_results['column'] == col]['value']
		print(col, relevant.mean(), relevant.values)

def quantify(filenames):
	file_results = get_numeric_values(filenames)
	results = pd.DataFrame(# add more data here, so we can use it in spreadsheet script
		[_get_info(filename, value) for filename, value in zip(filenames, file_results)],
		columns=["plate", "column", "well", "value"]
	)
	control_values = _calculate_control_values(results)
	return results.apply(lambda x: _normalize(x, control_values), axis=1)

def get_numeric_value(filename):
	with warnings.catch_warnings():
		warnings.simplefilter("ignore", UserWarning)
		img = imageio.imread(filename)[:,:,1]
	mask = imageops.get_fish_mask(filename.replace('CH1', 'CH4'), False, False)
	img = imageops.apply_mask(img, mask)
	total = img.sum(dtype=np.uint64, where=(img>5_000))
	return total if total > 0 else np.nan

def get_numeric_values(filenames, parallel=True):
	if parallel:
		return Pool(8).map(get_numeric_value, filenames)
	else:
		return map(get_numeric_value, filenames)

def _calculate_control_values(results):
	control_values = {}

	for plate in np.unique(results['plate']):
		plate_control_results = results[(results['plate'] == plate) & (results['column'] == 'B')]

		while True:
			control_values[plate] = float(plate_control_results.mean(numeric_only=True))
			upper = control_values[plate] * 1.5
			lower = control_values[plate] * 0.5

			if not plate_control_results[
				(plate_control_results['value'] > upper)
				| (plate_control_results['value'] < lower)
			].any(axis=None):
				break

			plate_control_results = plate_control_results[
				(plate_control_results['value'] <= upper)
				& (plate_control_results['value'] >= lower)
			]

	return control_values

def _get_info(filename, value):
	match = re.search(r'([a-zA-Z0-9]+)_XY([0-9][0-9])_', filename)
	if not match:
		raise ValueError('Filename %s missing needed xy information' % filename)

	plate = match.group(1)
	well = keyence.xy_to_well(int(match.group(2)))
	column = well[0]

	return (plate, column, well, value)

def _normalize(row, control_values):
	try:
		new_value = float(row['value'] * 100 // control_values[row['plate']])
		new_value = new_value if new_value < 150 else np.nan # discard results >=150% of control
	except ZeroDivisionError:
		print('ERROR: Plate', row['plate'], 'column', row['column'], 'with value',
			row['value'], 'has control value', control_values[row['plate']])
		new_value = np.nan

	return pd.Series([row['plate'], row['column'], row['well'], new_value], index=row.index)

if __name__ == '__main__':
	if len(sys.argv) > 1:
		main(sys.argv[1:])
	else:
		raise TypeError('Invoke with an argument, i.e. the name of a file or files to analyze.')
