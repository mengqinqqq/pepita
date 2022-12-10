import argparse
import json
import numpy as np
import os
import seaborn as sns
import sys
import warnings

import absolute
import analyze
import imageops
import pipeline
import util

LOG_DIR = f'{util.get_config("log_dir")}/dose_response'
ABS_MAX = int(util.get_config('absolute_max_infection'))
ABS_MIN = int(util.get_config('absolute_min_infection'))

replacement_delim = util.get_config('filename_replacement_delimiter')
replacement_brfld = util.get_config('filename_replacement_brightfield_infection').split(replacement_delim)
replacement_mask = util.get_config('filename_replacement_mask_infection').split(replacement_delim)
replacement_subtr = util.get_config('filename_replacement_subtr_infection').split(replacement_delim)

class InfectionImage(analyze.Image):
	channel = 0
	particles = False
	replacement_brfld = replacement_brfld
	replacement_mask = replacement_mask
	replacement_subtr = replacement_subtr

	def get_raw_value(self, threshold=0.02):
		if self.value is None:
			fl_img_masked = imageops.apply_mask(self.get_fl_img(), self.get_mask())
			max_value = imageops._get_bit_depth(fl_img_masked)[1]
			total = fl_img_masked.sum(
				dtype=np.uint64, where=(fl_img_masked > max_value*threshold))
			self.value = total if total > 0 else np.nan
		return self.value

def log(results):
	return {key: np.log2(values).tolist() for key, values in results.items()}

def main(imagefiles, cap=-1, chartfile=None, checkerboard=False, conversions=[], debug=0,
		platefile=None, plate_control=['B'], plate_info=None,
		plate_positive_control=[], treatment_platefile=None, absolute_chart=False, silent=False,
		talk=False):
	hashfile = util.get_inputs_hashfile(imagefiles=imagefiles, cap=cap, platefile=platefile,
		plate_control=plate_control)

	if talk:
		sns.set_context('talk')

	if debug == 0 and os.path.exists(hashfile):
		with open(hashfile, 'r') as f: # read cached results
			results = json.load(f)

		for group, relevant_values in results.items():
			if not silent:
				with warnings.catch_warnings():
					warnings.simplefilter('ignore', RuntimeWarning)
					print(group, np.nanmedian(relevant_values), relevant_values)
	else:
		results = quantify_infection(imagefiles=imagefiles, cap=cap, debug=debug,
			platefile=platefile, plate_control=plate_control, silent=False)
		with open(hashfile, 'w') as f: # cache results for reuse
			json.dump(results, f, ensure_ascii=False)

	if chartfile:
		analyze.chart(log(results), chartfile)

	conversions = dict(conversions)
	drug_conditions = _parse_results(results, conversions)
	control_drugs = [util.Cocktail(util.Dose(control).drug) for control in plate_control]
	models = {}

	results = {util.Solution(key, conversions): value for key, value in results.items()}

	# generate plate schematics

	schematic = analyze.get_schematic(platefile, len(imagefiles), flat=False)

	max_result = np.log2(max(val for vals_list in results.values() for val in vals_list))

	pipeline.generate_plate_schematic(schematic, log(results), conversions=conversions,
		plate_info=plate_info, well_count=96, cmap=sns.dark_palette('red', as_cmap=True),
		max_val=max_result)

def quantify_infection(imagefiles, cap=-1, debug=0, platefile=None, plate_control=['B'],
		silent=False):
	results = {}

	schematic = analyze.get_schematic(platefile, len(imagefiles))
	groups = list(dict.fromkeys(schematic))# deduplicated copy of `schematic`

	images = [InfectionImage(filename, group, debug) \
		for filename, group in zip(imagefiles, schematic)]

	for group in groups:
		relevant_values = [absolute.get_absolute_value(img) for img in images if img.group == group]
		results[group] = relevant_values
		if not silent:
			with warnings.catch_warnings():
				warnings.simplefilter('ignore', RuntimeWarning)
				print(group, np.nanmedian(relevant_values), relevant_values)

	return results

def _key_value_pair(argument, delimiter='='):
	return tuple(argument.split(delimiter))

def _parse_results(results, conversions):
	drug_conditions = {}
	for condition in results:
		solution = util.Solution(condition, conversions)
		util.put_multimap(drug_conditions, solution.get_cocktail(), solution)
	return drug_conditions

#
# main
#

if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description=('Analyzer for images of whole zebrafish with stained neuromasts, for the '
			'purposes of measuring hair cell damage under drug-combination conditions. Reports '
			'values relative to control.'))

	parser.add_argument('-cb', '--checkerboard',
		action='store_true',
		help=('If present, the input will be treated as a checkerboard assay, with output produced '
			'accordingly.'))

	parser.add_argument('-cv', '--conversions',
		default=[],
		nargs='*',
		type=_key_value_pair,
		help=('List of conversions between dose concentration labels and concrete values, each as '
			'a separate argument, each delimited by an equals sign. For instance, ABC50 might be '
			'an abbreviation for the EC50 of drug ABC, in which case the concrete concentration '
			'can be supplied like "ABC50=ABC 1mM" (make sure to quote, or escape spaces).'))

	parser.add_argument('-ppc', '--plate-positive-control',
		default=[],
		nargs='*',
		help=('Labels to treat as the positive control conditions in the plate schematic (i.e. '
			'conditions showing maximum effect). These wells are used to normalize all values in '
			'the plate for more interpretable results. Any number of values may be passed.'))

	parser.add_argument('--plate-info',
		default=None,
		help=('Any information identifying the plate(s) being analyzed that should be passed along '
			'to files created by this process.'))

	parser.add_argument('-tp', '--treatment-platefile',
		help='CSV file containing a schematic of the plate in which the imaged fish were treated. '
			'Used to chart responses by treatment location, if desired. Row and column headers are '
			'optional. The cell values are essentially just arbitrary labels: results will be '
			'grouped and charted according to the supplied values.')

	parser.add_argument('--absolute-chart',
		action='store_true',
		help=('If present, a plate graphic will be generated with absolute (rather than relative) '
			'brightness values.'))

	parser.add_argument('--talk',
		action='store_true',
		help=('If present, images will be generated with the Seaborn "talk" context.'))

	analyze.set_arguments(parser)

	util.remove_arguments(parser, 'plate_ignore', 'group_regex')

	args = parser.parse_args(sys.argv[1:])
	args_dict = vars(args)
	try:
		main(**args_dict)
	except analyze.UserError as ue:
		print('Error:', ue)
		sys.exit(1)
