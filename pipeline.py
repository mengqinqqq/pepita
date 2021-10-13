import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import seaborn as sns
import sys
import warnings

import analyze
import dose_response
import util

def main(imagefiles, cap=150, chartfile=None, debug=0, group_regex='.*', platefile=None,
		plate_control=['B'], plate_ignore=[], silent=False):
	results = analyze.main(imagefiles, cap, chartfile, debug, group_regex, platefile,
		plate_control, plate_ignore, True)

	drug_conditions = _parse_results(results)
	control_drugs = [(util.Dose(control).drug,) for control in plate_control]
	models = {}

	for drug, conditions in drug_conditions.items():
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
		models[drug] = dose_response.Model(
			conditions, scores, drug, E_max=dose_response.neo_E_max(), debug=1)

	model_combo = next(model for model in models.values() if model.combo)
	model_a = models[(model_combo.condition[0],)]
	model_b = models[(model_combo.condition[1],)]
	dose_response.chart_pair(model_a, model_b, model_combo)
	combo_FIC_50 = dose_response.get_combo_FIC(0.5, model_a, model_b, model_combo)
	combo_FIC_75 = dose_response.get_combo_FIC(0.75, model_a, model_b, model_combo)
	print(f'{model_combo.get_condition()}: FIC_50 {combo_FIC_50}, FIC_75 {combo_FIC_75}')
	print((
		f'EC_50: {model_a.get_condition()}={model_a.effective_concentration(0.5)}, '
		f'{model_b.get_condition()}={model_b.effective_concentration(0.5)}, '
		f'{model_combo.get_condition()}={model_combo.effective_concentration(0.5)}'
	))
	print((
		f'EC_75: {model_a.get_condition()}={model_a.effective_concentration(0.75)}, '
		f'{model_b.get_condition()}={model_b.effective_concentration(0.75)}, '
		f'{model_combo.get_condition()}={model_combo.effective_concentration(0.75)}'
	))
	print((
		f'EC_90: {model_a.get_condition()}={model_a.effective_concentration(0.9)}, '
		f'{model_b.get_condition()}={model_b.effective_concentration(0.9)}, '
		f'{model_combo.get_condition()}={model_combo.effective_concentration(0.9)}'
	))

def _parse_results(results):
	drug_conditions = {}
	for condition in results:
		solution = util.Solution(condition)
		util.put_multimap(drug_conditions, solution.get_drugs(), solution)
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
