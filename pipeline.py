import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
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


	sns.set_theme(style="darkgrid")

	data = pd.DataFrame({
		'brightness': [value for values in results.values() for value in values],
		'concentration': [util.extract_number(key) for key, values in results.items() for _ in values],
	})
	ax = sns.scatterplot(
		x='concentration', y='brightness', data=data, color='black', label='Measurements',
		marker='.', s=64)

	model = next(iter(models.values()))
	plt.scatter(model.xs, model.ys, color='black', marker='_', s=256)

	line_xs = np.linspace(0, float(max(model.xs)), 100)
	line_ys = model.get_ys(line_xs)
	sns.lineplot(x=line_xs, y=line_ys, ax=ax, label='Model')
	plt.scatter(
		model.effective_concentration(0.5), 4, color='black', label='EC50', marker='|', s=128)

	ax.set_ylim(bottom=0)
	plt.title(f'{model.get_condition()} Dose-Response Curve')
	plt.xlabel(f'{model.get_condition()} Dose (μM)')
	plt.ylabel('Pipeline Score')
	plt.legend()

	plt.savefig(
		('/home/ethan/Dropbox/Ethan/Project INDIGO-Tox/progress/RPPR_2021Q4/figures/'
		f'pipeline_components/{model.get_condition()}.png'))
	plt.close()
	plt.clf()

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
