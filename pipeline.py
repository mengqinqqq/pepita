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

def main(imagefiles, cap=150, chartfile=None, debug=0, group_regex='.*', platefile=None,
		plate_control=['B'], plate_ignore=[], silent=False):
	results = analyze.main(imagefiles, cap, chartfile, debug, group_regex, platefile,
		plate_control, plate_ignore, True)

	drug_conditions = _parse_results(results)
	models = []

	for drug, subconditions in drug_conditions.items():
		if len(subconditions) <= 1:
			print(drug, 'skipped:', subconditions)
			continue
		doses = []
		scores = []
		for dose, unit in subconditions:
			with warnings.catch_warnings():
				warnings.simplefilter('ignore', RuntimeWarning)
				summary_score = np.nanmedian(results[condition2key(drug, dose, unit)])
				if np.isnan(summary_score):
					continue
				scores.append(summary_score)
			doses.append(_string2float(dose))
		print(drug, doses, scores)
		model = dose_response.Model(doses, scores, drug, E_max=dose_response.neo_E_max(), debug=1)
		models.append(model)

	model_combo = next(model for model in models if model.combo)
	model_a = next(model for model in models if model.condition == model_combo.condition[0])
	model_b = next(model for model in models if model.condition == model_combo.condition[1])
	dose_response.chart_pair(model_a, model_b, model_combo)
	combo_FIC_50 = dose_response.get_combo_FIC(
		0.5, model_a, model_b, model_combo, model_combo.proportion_a)
	combo_FIC_75 = dose_response.get_combo_FIC(
		0.75, model_a, model_b, model_combo, model_combo.proportion_a)
	print(f'{model_combo.get_condition()}: FIC_50 {combo_FIC_50}, FIC_75 {combo_FIC_75}')

def condition2key(drug, dose, unit):
	if isinstance(drug, str): # single drug
		return f'{drug} {dose}{unit}'
	else: # drug combo
		return f'{drug[0]} {dose[0]}{unit[0]} + {drug[1]} {dose[1]}{unit[1]}'

def _condition_error(condition):
	return f'Condition {condition} is not in the proper format. Conditions should be ' +\
		'represented as "[drug] [dose][units]". Combinations should be two such strings joined ' +\
			'with a " + ".'

def _parse_results(results):
	condition_pattern = re.compile(r'(.+?) ([0-9]+[.]?[0-9]*)([^0-9 ]+)')
	drug_conditions = {}

	for condition in results:
		conditions = condition.split(' + ')
		if len(conditions) == 1: # single drug
			match = condition_pattern.match(conditions[0])
			if not match:
				raise analyze.UserError(_condition_error(condition))
			drug, dose, unit = match.group(1, 2, 3)
			_put(drug_conditions, drug, (dose, unit))
		else: # drug combo
			match1 = condition_pattern.match(conditions[0])
			if not match1:
				raise analyze.UserError(_condition_error(condition))
			match2 = condition_pattern.match(conditions[1])
			if not match2:
				raise analyze.UserError(_condition_error(condition))
			drug1, dose1, unit1 = match1.group(1, 2, 3)
			drug2, dose2, unit2 = match2.group(1, 2, 3)
			_put(drug_conditions, (drug1, drug2), ((dose1, dose2), (unit1, unit2)))
	return drug_conditions

def _put(dict_, key, value):
	list_ = dict_.get(key, [])
	list_.append(value)
	dict_[key] = list_

def _string2float(string_or_iterable):
	if isinstance(string_or_iterable, str):
		return float(string_or_iterable)
	else:
		return tuple(float(s) for s in string_or_iterable)

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
