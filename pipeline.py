import argparse
import json
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import warnings

import analyze
import dose_response
import interactions2
import util

def main(imagefiles, cap=-1, chartfile=None, checkerboard=False, conversions=[], debug=0,
		group_regex='.*', platefile=None, plate_control=['B'], plate_ignore=[], plate_info=None,
		plate_positive_control=[], silent=False):
	hashfile = util.get_inputs_hashfile(imagefiles=imagefiles, cap=cap, group_regex=group_regex,
		platefile=platefile, plate_control=plate_control, plate_ignore=plate_ignore)

	if chartfile is None and debug == 0 and os.path.exists(hashfile):
		with open(hashfile, 'r') as f: # read cached results
			results = json.load(f)
	else:
		results = analyze.main(imagefiles, cap, chartfile, debug, group_regex, platefile,
			plate_control, plate_ignore, True)
		with open(hashfile, 'w') as f: # cache results for reuse
			json.dump(results, f, ensure_ascii=False)

	conversions = dict(conversions)
	drug_conditions = _parse_results(results, conversions)
	control_drugs = [util.Cocktail(util.Dose(control).drug) for control in plate_control]
	models = {}

	results = {str(util.Solution(key, conversions)): value for key, value in results.items()}

	# positive control

	positive_control_solutions = [
		util.Solution(positive_control, conversions) for positive_control in plate_positive_control]
	positive_control_scores = [
		result for solution in positive_control_solutions for result in results[str(solution)]]
	with warnings.catch_warnings():
		warnings.simplefilter('ignore', RuntimeWarning)
		positive_control_value = np.nanmean(positive_control_scores)

	if np.isnan(positive_control_value):
		print(('WARNING: No positive control included. Using minimum calculated value as '
			'positive control'))
		positive_control_value = np.nanmin(
			[value for condition, values in results.items() for value in values])

	# generate models, dose-response charts

	for cocktail, conditions in drug_conditions.items():
		if cocktail.drugs == ('Control',):
			continue
		cocktail_scores = {}
		summary_scores = []
		for control_drug in control_drugs:
			for solution in drug_conditions[control_drug]:
				conditions.insert(0, solution)
		for solution in conditions:
			cocktail_scores[solution] = results[str(solution)]
			with warnings.catch_warnings():
				warnings.simplefilter('ignore', RuntimeWarning)
				summary_score = np.nanmedian(results[str(solution)])
				summary_scores.append(summary_score)
		models[cocktail] = dose_response.Model(
			conditions, summary_scores, cocktail, E_max=positive_control_value)
		models[cocktail].chart(results[str(solution)], datapoints=cocktail_scores,
			name=plate_info + '_' + str(cocktail) if plate_info else None,
			scale=[positive_control_value, 100])

	# print EC values

	for model in models.values():
		for ec_value in (50, 75, 90):
			concentn = model.effective_concentration(ec_value / 100)
			if not np.isnan(concentn):
				print((f'{model.get_condition()} '
					f'EC_{ec_value}={concentn:.2f}{model.get_x_units()}'))

	# analyze combinations

	models_combo = [model for model in models.values() if model.combo]

	if not checkerboard:
		total_max_x = 1
		total_max_y = 1

		fig = plt.figure()
		fig.set_size_inches(12, 8)
		fig.set_dpi(100)
		ax = fig.add_subplot(1, 1, 1)
		ax.margins(0.006)

		for model_combo in models_combo:
			subcocktail_a = util.Cocktail(model_combo.cocktail.drugs[0])
			if subcocktail_a not in models:
				continue
			subcocktail_b = util.Cocktail(model_combo.cocktail.drugs[1])
			model_a = models[subcocktail_a]
			model_b = models[subcocktail_b]
			plot_filename, max_x, max_y = dose_response.analyze_diamond(
				model_a, model_b, model_combo)
			dose_response.chart_diamond(model_a, model_b, model_combo)

			total_max_x = max(total_max_x, max_x)
			total_max_y = max(total_max_y, max_y)

		if models_combo:
			plt.xlim(right=total_max_x)
			plt.ylim(top=total_max_y)
			plt.savefig(plot_filename)
			plt.close()
			plt.clf()
	else:
		model_combo_0 = models_combo[0]
		model_a = models[util.Cocktail(model_combo_0.cocktail.drugs[0])]
		model_b = models[util.Cocktail(model_combo_0.cocktail.drugs[1])]

		# sns.set_context('talk')

		dose_response.analyze_checkerboard(model_a, model_b, models_combo, method='Bliss',
			file_name_context=plate_info)
		dose_response.chart_checkerboard(model_a, model_b, models_combo,
			file_name_context=plate_info)

		doses_a = np.array([x.doses[0] for x in model_a.xs if x.get_drugs() != ('Control',)])
		doses_b = np.array([x.doses[0] for x in model_b.xs if x.get_drugs() != ('Control',)])

		responses_all_a = squarify(
			[results[str(x)] for x in model_a.xs if x.get_drugs() != ('Control',)])
		responses_all_b = squarify(
			[results[str(x)] for x in model_b.xs if x.get_drugs() != ('Control',)])

		combo_results = results.copy()
		[combo_results.pop(str(solution), None) for solution in model_a.xs]
		[combo_results.pop(str(solution), None) for solution in model_b.xs]
		[combo_results.pop(str(solution), None) for solution in positive_control_solutions]
		combo_solutions = [util.Solution(condition, conversions) for condition in combo_results]

		doses_a_ab = np.array([solution.doses[0] for solution in combo_solutions])
		doses_b_ab = np.array([solution.doses[1] for solution in combo_solutions])
		responses_all_ab = squarify([results[str(solution)] for solution in combo_solutions])

		if not positive_control_scores:
			positive_control_scores = np.array([
				min([result for results_list in results.values() for result in results_list])
			])

		if not plate_info:
			plate_info = os.path.basename(os.path.dirname(os.path.dirname(imagefiles[0])))

		interactions2.response_surface(doses_a, responses_all_a, doses_b, responses_all_b,
			doses_a_ab, doses_b_ab, responses_all_ab, positive_control_scores,
			sampling_iterations=1000, sample_size=20, model_size=1, alpha=0.1,
			file_name_context=plate_info)

def squarify(list_of_lists):
	width = max([len(row) for row in list_of_lists])

	for row in list_of_lists:
		pad_size = width - len(row)
		row.extend([np.nan for _ in range(pad_size)])

	return np.array(list_of_lists)

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

	analyze.set_arguments(parser)

	args = parser.parse_args(sys.argv[1:])
	args_dict = vars(args)
	try:
		main(**args_dict)
	except analyze.UserError as ue:
		print('Error:', ue)
		sys.exit(1)
