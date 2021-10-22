import argparse
import json
import numpy as np
import os
import sys
import warnings

import analyze
import dose_response
import util

def main(imagefiles, cap=150, chartfile=None, debug=0, group_regex='.*', platefile=None,
		plate_control=['B'], plate_ignore=[], silent=False):
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

	drug_conditions = _parse_results(results)
	control_drugs = [util.Cocktail(util.Dose(control).drug) for control in plate_control]
	models = {}

	for cocktail, conditions in drug_conditions.items():
		if len(conditions) < 3: # can't create a proper model with less than 3 datapoints
			continue
		scores = []
		for control_drug in control_drugs:
			for solution in drug_conditions[control_drug]:
				conditions.insert(0, solution)
		for solution in conditions:
			with warnings.catch_warnings():
				warnings.simplefilter('ignore', RuntimeWarning)
				summary_score = np.nanmedian(results[solution.string])
				scores.append(summary_score)
		models[cocktail] = dose_response.Model(
			conditions, scores, cocktail, E_max=dose_response.neo_E_max(), debug=1)

	for model in models.values():
		for ec_value in (10, 25, 50, 75, 90):
			concentn = model.effective_concentration(ec_value / 100)
			if not np.isnan(concentn):
				print(f'{model.get_condition()}: EC_{ec_value}={concentn:.2f}{model.get_x_units()}')

	models_combo = [model for model in models.values() if model.combo]
	for model_combo in models_combo:
		subcocktail_a = util.Cocktail(model_combo.cocktail.drugs[0])
		if subcocktail_a not in models:
			continue
		subcocktail_b = util.Cocktail(model_combo.cocktail.drugs[1])
		model_a = models[subcocktail_a]
		model_b = models[subcocktail_b]
		dose_response.chart_pair(model_a, model_b, model_combo)
		combo_FIC_50 = dose_response.get_combo_FIC(0.5, model_a, model_b, model_combo)
		combo_FIC_75 = dose_response.get_combo_FIC(0.75, model_a, model_b, model_combo)
		print(f'{model_combo.get_condition()}: FIC_50 {combo_FIC_50}, FIC_75 {combo_FIC_75}')

def _parse_results(results):
	drug_conditions = {}
	for condition in results:
		solution = util.Solution(condition)
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

	analyze.set_arguments(parser)

	args = parser.parse_args(sys.argv[1:])
	args_dict = vars(args)
	try:
		main(**args_dict)
	except analyze.UserError as ue:
		print('Error:', ue)
		sys.exit(1)
