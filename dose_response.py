import csv
from scipy.optimize import curve_fit, OptimizeWarning
import numpy as np
import os.path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from time import time
import warnings

import util

LOG_DIR = f'{util.get_config("log_dir")}/dose_response'
_neo_model = None

class Model:
	def __init__(self, xs, ys, condition, E_0=100, E_max=None, debug=0):
		self.combo = not isinstance(condition, str) and len(condition) > 1
		self.condition = condition
		self.xs = xs
		self.ys = ys
		self.E_0 = E_0
		self.E_max = E_max

		self.equation = lambda xs, b, c, e: log_logistic_model(xs, b, c, E_0, e)
		with warnings.catch_warnings():
			warnings.simplefilter('ignore', RuntimeWarning)
			warnings.simplefilter('ignore', OptimizeWarning)
			popt, pcov = curve_fit(self.equation, self.xs, self.ys)
		self.b, self.c, self.e = popt

		if debug > 0:
			self.chart()

	def __repr__(self):
		return "{}({})".format(self.__class__.__name__, self.condition)

	def chart(self):
		plt.scatter(self.xs, self.ys, color='black', label='Data', marker='.')
		plt.xlabel(f'{self.get_condition()} Dose (μM)')
		plt.ylabel('Pipeline Score')

		max_x = max(self.xs)
		line_xs = np.linspace(0, max_x, 100)
		line_ys = self.get_ys(line_xs)
		plt.plot(line_xs, line_ys, color='darkgrey', label='Model')
		plt.plot(line_xs, np.ones_like(line_xs) * self.E_0, color='lightgrey', label='E_0')
		plt.plot(line_xs, np.ones_like(line_xs) * self.get_condition_E_max(),
			color='lightgrey', label='E_max')
		plt.plot(line_xs, np.ones_like(line_xs) * self.get_absolute_E_max(),
			color='lightgrey', label='Abs_max')
		ec_50 = self.effective_concentration(0.5)
		plt.scatter(ec_50, self.get_ys(ec_50), color='black', label='EC_50', marker='+')

		plt.legend()
		unique_str = str(int(time() * 1000) % 1_620_000_000_000)
		plt.savefig(f'{LOG_DIR}/{self.get_condition()}_{unique_str}.png')
		plt.close()
		plt.clf()

	# pct_survival = (f(x) - min) / (max - min)
	# f(x) = c + (d - c) / (1 + (x / e)**b)
	# yields x
	def effective_concentration(self, pct_inhibition):
		if pct_inhibition <= 0 or pct_inhibition >= 1:
			raise RuntimeError('Inhibition level must be between 0 and 1')

		b = self.b
		c = self.c
		d = self.E_0
		e = self.e
		max_ = self.E_0
		min_ = self.get_absolute_E_max()
		pct_survival = 1 - pct_inhibition

		pct_pts_above_E_max = pct_survival * (max_ - min_) + min_ - c

		if pct_pts_above_E_max < 1:
			print(f'WARN: {self.get_condition()} EC_{int(pct_inhibition*100)} is unreachable')
			return np.nan

		return e * ((d - c)/pct_pts_above_E_max - 1)**(1/b)

	def get_absolute_E_max(self):
		return self.E_max if self.E_max is not None else self.c

	def get_condition(self):
		if isinstance(self.condition, str):
			return self.condition
		else:
			return '+'.join(self.condition)

	def get_condition_E_max(self):
		return self.c

	def get_ys(self, xs):
		return self.equation(xs, self.b, self.c, self.e)

	def __repr__(self):
		return str(self.__dict__)

def chart_pair(model_a, model_b, model_combo):
	# data chart

	data = pd.DataFrame({
		'concentration': list(model_combo.xs) * 3,
		'score': list(model_a.ys) + list(model_b.ys) + list(model_combo.ys),
		'condition': [model_a.get_condition()] * len(model_a.xs)
			+ [model_b.get_condition()] * len(model_b.xs)
			+ [model_combo.get_condition()] * len(model_combo.xs)
	})
	data = data.pivot_table(
		index='condition', columns='concentration', values='score', aggfunc=np.nanmean)

	sns.heatmap(data,
		vmin=model_a.get_absolute_E_max(), vmax=model_a.E_0, cmap='viridis', annot=True, fmt='.1f',
		linewidths=1, square=True)
	unique_str = str(int(time() * 1000) % 1_620_000_000_000)
	plt.title(f'{model_a.get_condition()} vs. {model_b.get_condition()}: Raw Data')
	plt.savefig(f'{LOG_DIR}/combo_{model_a.get_condition()}-{model_b.get_condition()}_data_{unique_str}.png')
	plt.clf()

	# model chart

	max_x = int(max(max(model_a.xs), max(model_b.xs)))
	max_x = max_x + 5 - (max_x % 5) # round up to the nearest 5
	xs = list(range(0, int(max_x), int(max_x / 5)))
	data = pd.DataFrame({
		'concentration': xs * 3,
		'score': np.concatenate((model_a.get_ys(xs), model_b.get_ys(xs), model_combo.get_ys(xs))),
		'condition': [model_a.get_condition()] * len(xs)
			+ [model_b.get_condition()] * len(xs)
			+ [model_combo.get_condition()] * len(xs)
	})
	data = data.pivot_table(
		index='condition', columns='concentration', values='score', aggfunc='median')

	sns.heatmap(data,
		vmin=model_a.get_absolute_E_max(), vmax=model_a.E_0, cmap='viridis', annot=True, fmt='.1f',
		linewidths=1, square=True)
	plt.title(f'{model_a.get_condition()} vs. {model_b.get_condition()}: Model')
	plt.savefig(f'{LOG_DIR}/combo_{model_a.get_condition()}-{model_b.get_condition()}_model_{unique_str}.png')
	plt.clf()

def get_combo_FIC(pct_inhibition, model_a, model_b, model_combo):
	# set model_b to the model with the higher maximum effect = lower survival at maximum effect
	combo_proportion_a = model_combo.xs[-1].ratio()
	if model_a.c < model_b.c:
		model_a, model_b = model_b, model_a
		combo_proportion_a = combo_proportion_a.reciprocal()

	ec_b_alone = model_b.effective_concentration(pct_inhibition)
	ec_combo = model_combo.effective_concentration(pct_inhibition)

	if np.isnan(ec_b_alone) or np.isnan(ec_combo):
		return np.nan

	ec_combo_a = ec_combo * combo_proportion_a
	ec_combo_b = ec_combo * (1 - combo_proportion_a)
	inhibition_max_a = model_a.E_0 - model_a.c
	inhibition_max_b = model_b.E_0 - model_b.c

	return do_FIC(ec_combo_a, ec_combo_b, model_a.e, model_b.e, inhibition_max_a, inhibition_max_b,
		ec_b_alone, model_b.b, model_a.b)

# derived from Grabovsky and Tallarida 2004, http://doi.org/10.1124/jpet.104.067264, Eq. 3
# where B is the drug with the higher maximum effect = lower survival at maximum effect
# a_i = amount of A in combination required to reach relevant effect level
# b_i = amount of B in combination required to reach relevant effect level
# A_E50_a = amount of A alone required to reach 50% of A's maximum effect
# B_E50_b = amount of B alone required to reach 50% of B's maximum effect
# E_max_a = A's maximum effect
# E_max_b = B's maximum effect
# B_i = amount of B alone required to reach relevant effect level
# p = Hill function coefficient for B's dose-response curve
# q = Hill function coefficient for A's dose-response curve
def do_FIC(a_i, b_i, A_E50_a, B_E50_b, E_max_a, E_max_b, B_i, p, q):
	return (b_i + B_E50_b/((E_max_b/E_max_a)*(1 + A_E50_a**q/a_i**q) - 1)**(1/p)) / B_i

# Ritz 2009, https://doi.org/10.1002/etc.7, Eq. 2
# `xs` is a numpy array of x values; b, c, d, and e are model parameters:
# relative slope at inflection point, lower asymptote, upper asymptote, inflection point (EC_50)
# returns y values
def log_logistic_model(xs, b, c, d, e):
	return c + (d - c) / (1 + (xs / e)**b)

def neo_E_max():
	neo_model = _get_neo_model()
	return _neo_model.get_condition_E_max()

def _get_neo_model():
	global _neo_model
	if _neo_model == None:
		xs, ys = [], []

		with open(os.path.join(util.get_here(), 'examples/neo_data.csv'),
				encoding='utf8', newline='') as f:
			for x, y in csv.reader(f, delimiter='\t'):
				xs.append(util.Solution(f'Neomycin {x}μM'))
				ys.append(float(y))

		_neo_model = Model(xs, ys, 'Neomycin', debug=1)

	return _neo_model

#
# main
#

if __name__ == '__main__':
	model = _get_neo_model()

	ec_90 = model.effective_concentration(0.9)
	ec_75 = model.effective_concentration(0.75)
	ec_50 = model.effective_concentration(0.5)

	print(f'E_max: {model.get_absolute_E_max()} score')
	print(f'EC_90: {ec_90} μM')
	print(f'ec_75: {ec_75} μM')
	print(f'ec_50: {ec_50} μM')

	model_a = Model(model.xs, model.ys, ('Neo1'))
	model_b = Model(model.xs, model.ys, ('Neo2'))
	model_combo = Model(
		[x.dilute(0.5).combine_doses(x.dilute(0.5)) for x in model.xs], model.ys, ('Neo1', 'Neo2'))

	chart_pair(model_a, model_b, model_combo)
	neo_neo_FIC_50 = get_combo_FIC(0.5, model_a, model_b, model_combo)
	neo_neo_FIC_90 = get_combo_FIC(0.9, model_a, model_b, model_combo)
	print(f'Neomycin self-combo FIC_50={neo_neo_FIC_50}, FIC_90={neo_neo_FIC_90}')
