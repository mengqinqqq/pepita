# requires: pip install imagecodecs

import imageio
import json
import numpy as np
import pandas as pd
import re
import sys

import keyence

def quantify(filenames):
	# calculate results

	file_results = do_quantify3(filenames)

	results = pd.DataFrame([_get_info(filename, value) for filename, value in file_results.items()], columns=["plate", "column", "value"])

	# normalize results

	control_values = {}

	for plate in np.unique(results['plate']):
		plate_control_results = results[(results['plate'] == plate) & (results['column'] == 'B')]

		while True:
			control_values[plate] = float(plate_control_results.mean(numeric_only=True))
			threshold = control_values[plate] * 1.5

			if plate_control_results[plate_control_results['value'] > threshold].any(axis=None):
				plate_control_results = plate_control_results[plate_control_results['value'] <= threshold]
			else:
				break

	normalized_results = results.apply(lambda x: _normalize(x, control_values), axis=1)

	# print results

	for col in keyence.COLUMNS:
		relevant = normalized_results[normalized_results['column'] == col]['value']
		print(col, relevant.mean(), relevant.values)

def do_quantify3(filenames):
	results = {}

	for filename in filenames:
		img_array = imageio.imread(filename)
		results[filename] = img_array.sum(dtype=np.uint64, where=(img_array>5_000)) / img_array.size

	return results

def _get_info(filename, value):
	match = re.search(r'([a-zA-Z0-9]+)_XY([0-9][0-9])_', filename)
	if not match:
		raise ValueError('Filename %s missing needed information' % filename)

	plate = match.group(1)
	column = keyence.xy_to_well(int(match.group(2)))[0]

	return (plate, column, value)

def _normalize(entry, control_values):
	new_value = float(entry['value'] * 100 // control_values[entry['plate']])
	new_value = new_value if new_value < 150 else np.nan # discard results greater than 150% of control value
	return pd.Series([entry['plate'], entry['column'], new_value], index=entry.index)

if __name__ == '__main__':
	if len(sys.argv) > 1:
		quantify(sys.argv[1:])
	else:
		raise TypeError('Invoke with an argument, i.e. the name of a file to analyze.')
