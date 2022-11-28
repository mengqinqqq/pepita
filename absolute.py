import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import seaborn as sns
import sys
import warnings

import analyze

def chart(results, chartfile):
	with sns.axes_style(style='whitegrid'):
		data = pd.DataFrame({
			'brightness': [value for values in results.values() for value in values],
			'plate': [key for key, values in results.items() for _ in values],
		})

		ax = sns.swarmplot(x='plate', y='brightness', data=data, size=3)
		ax.set_ylim(bottom=0)
		sns.boxplot(x='plate', y='brightness', data=data, showbox=False, showcaps=False,
			showfliers=False, whiskerprops={'visible': False})
		plt.xticks(rotation=80)
		plt.tight_layout()
		plt.savefig(chartfile)

# from ISO 12232:1998 via https://en.wikipedia.org/wiki/Film_speed#Measurements_and_calculations
# H = qLt/(N^2)
# H: luminous exposure (lux-seconds), proportional to pixel value
# L: luminance = variable of interest (candela/m^2)
# t: exposure time (seconds)
# N: aperture f-stop (unitless)
# q = (π/4) * T * v(θ) * cos^4(θ)
# T: transmittance of lens system (should be roughly constant for a given microscope)
# v: vignetting factor (currently ignored, could maybe be calculated from brightfield image)
# θ: angle relative to the lens
# therefore solved for L: L = H * N^2 / qt
# from https://en.wikipedia.org/wiki/Numerical_aperture#Numerical_aperture_versus_f-number
# N = f/D; NA = n * sin(arctan(D/2f))
# f: focal length of optical system
# D: diameter of entrance pupil
# NA: numerical aperture of the lens
# n: index of refraction of working medium (1.00 for air, 1.33 for water, 1.52 for immersion oil)
# therefore N in terms of NA and n: N = 1 / (2 * tan(arcsin(NA/n)))
# values are returned in units proportional to candela/m^2
def get_absolute_value(image, debug=0, n=1.0, transmittance=1, vignette=lambda theta: 1):
	metadata = image.get_fl_metadata()

	H = image.get_raw_value()
	N = 1 / (2 * math.tan(math.asin(metadata['Numerical Aperture'] / n)))
	theta = 0 # TODO: calculate theta from working distance & weighted average(?) of signal in img
	q = (math.pi / 4) * transmittance * vignette(theta) * math.cos(theta)**4
	t = metadata['Exposure']['Value']

	if debug >= 1:
		print('%s: H=%f "lx⋅s", N=%f, theta=%f rad, q=%f, t=%fs' %
				(image.fl_filename, H, N, theta, q, t))

	L = H * N**2 / q / t

	return np.nan if np.isnan(L) else int(L / 10_000) # to make results more readable, "cd/cm^2"

def main(imagefiles, cap=-1, chartfile=None, debug=0, group_regex='.*', platefile=None,
		plate_control=['B'], plate_ignore=[], silent=False):
	results = {}

	schematic = analyze.get_schematic(platefile, len(imagefiles), plate_ignore)
	groups = list(dict.fromkeys(schematic))# deduplicated copy of `schematic`
	pattern = re.compile(group_regex)
	images = [analyze.Image(filename, group, debug) \
		for filename, group in zip(imagefiles, schematic) \
			if group in plate_control or pattern.search(group)]

	pattern = re.compile(group_regex)
	for group in groups:
		if group in plate_control or pattern.search(group):
			relevant_values = [get_absolute_value(img) for img in images if img.group == group]
			results[group] = relevant_values
			if not silent:
				with warnings.catch_warnings():
					warnings.simplefilter('ignore', RuntimeWarning)
					print(group, np.nanmedian(relevant_values), relevant_values)

	if chartfile:
		chart(results, chartfile)

	return results

#
# main
#

if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description=('Analyzer for images of whole zebrafish with stained neuromasts, for the '
			'purposes of measuring hair cell damage in absolute terms. Reports values in '
			'arbitrary units not relative to any other value.'))

	analyze.set_arguments(parser)

	args = parser.parse_args(sys.argv[1:])
	args_dict = vars(args)
	main(**args_dict)
