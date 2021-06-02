# requires: pip install imagecodecs

import imageio
import json
import numpy as np
import re
import sys

def quantify(filenames):
	file_results = do_quantify1(filenames)
	column_results = {}

	for filename, value in file_results.items():
		match = re.search(r'_XY([0-9][0-9])_', filename)
		if not match:
			raise ValueError('Filename missing needed information')

		xy = int(match.group(1))
		column = ['B', 'C', 'D', 'E', 'F', 'G'][(xy-1) // 10]

		column_results[column] = column_results.get(column, [])
		column_results[column].append(value)

	for column, results in column_results.items():
		results = np.array(results)
		valid_results = results.compress(np.isfinite(results))
		print(column, valid_results.mean(), results)

def do_quantify1(filenames):
	results = {}

	for filename in filenames:
		img_array = imageio.imread(filename)
		value = img_array.clip(min=2**10).sum(dtype=np.uint64)
		results[filename] = value // 2**22 - 2020

	return results

def do_quantify2(filenames):
	results = {}

	for filename in filenames:
		img_array = imageio.imread(filename)
		value = img_array.sum(dtype=np.uint64, where=(img_array>5_000)) // img_array.size
		results[filename] = value if value < 100 else np.nan

	return results

if __name__ == '__main__':
	if len(sys.argv) > 1:
		quantify(sys.argv[1:])
	else:
		raise TypeError('Invoke with an argument, i.e. the name of a file or files to analyze.')
