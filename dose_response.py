import csv
import numpy as np
import os.path
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize
import seaborn as sns
import sys
from time import time
import warnings

import util

LOG_DIR = f'{util.get_config("log_dir")}/dose_response'
_neo_model = None

class Model:
	def __init__(self, xs, ys, cocktail, E_0=100, E_max=None, debug=0):
		self.cocktail = cocktail
		self.combo = len(cocktail.drugs) > 1
		self.xs = xs
		self.ys = ys
		self.E_0 = E_0
		self.E_max = E_max

		self.equation = lambda xs, b, c, e: log_logistic_model(xs, b, c, E_0, e)
		with warnings.catch_warnings():
			warnings.simplefilter('ignore', RuntimeWarning)
			warnings.simplefilter('ignore', scipy.optimize.OptimizeWarning)
			popt, pcov = scipy.optimize.curve_fit(self.equation, self.xs, self.ys)
		self.b, self.c, self.e = popt

		if debug > 0:
			self.chart()

	def __repr__(self):
		return "{}({})".format(self.__class__.__name__, self.cocktail)

	def chart(self, close=True, color='darkgrey', label=True, name=None):
		plt.scatter(self.xs, self.ys, color='black', label='Data', marker='.')
		if label:
			plt.xlabel(f'{self.get_condition()} Dose (μM)')
			plt.ylabel('Pipeline Score')

		line_xs = np.linspace(0, float(max(self.xs)), 100)
		line_ys = self.get_ys(line_xs)
		plt.plot(line_xs, line_ys, color=color, label='Model')
		plt.plot(line_xs, np.ones_like(line_xs) * self.E_0, color='lightgrey', label='E_0')
		plt.plot(line_xs, np.ones_like(line_xs) * self.get_condition_E_max(),
			color='lightgrey', label='E_max')
		plt.plot(line_xs, np.ones_like(line_xs) * self.get_absolute_E_max(),
			color='lightgrey', label='Abs_max')
		ec_50 = self.effective_concentration(0.5)
		plt.scatter(ec_50, self.get_ys(ec_50), color='black', label='EC_50', marker='+')

		if label:
			plt.legend()

		if close:
			uniq_str = str(int(time() * 1000) % 1_620_000_000_000)
			if not name:
				name = self.get_condition()
			plt.savefig(os.path.join(LOG_DIR, f'{name}_{uniq_str}.png'))
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

		if pct_pts_above_E_max <= 0:
			print(f'WARN: {self.get_condition()} EC_{int(pct_inhibition*100)} is unreachable')
			return np.nan

		return e * ((d - c)/pct_pts_above_E_max - 1)**(1/b)

	def get_absolute_E_max(self):
		return self.E_max if self.E_max is not None else self.c

	def get_condition(self):
		return str(self.cocktail)

	def get_condition_E_max(self):
		return self.c

	def get_intersection(self, other, guess, ratio):
		if not isinstance(other, Model):
			return np.nan

		other_adj = Model(np.array(other.xs) * ratio, other.ys, other.cocktail, other.E_0,
			other.E_max)

		f_intersection_equals_zero = lambda xs: np.array(
			self.equation(xs, self.b, self.c, self.e) \
				- other_adj.equation(xs, other_adj.b, other_adj.c, other_adj.e),
			dtype=np.float64)

		plot_func(self.xs, f_intersection_equals_zero, f'{self.cocktail} - {other_adj.cocktail}',
			f'a_b_intersection@{ratio}', f'{self.cocktail} ({self.get_x_units()})')

		return get_intersection(
			lambda xs: self.equation(xs, self.b, self.c, self.e),
			lambda xs: other_adj.equation(xs, other_adj.b, other_adj.c, other_adj.e),
			guess)

	# pct_survival = (f(x) - min) / (max - min)
	def get_pct_survival(self, xs=None, ys=None):
		if not xs and not ys:
			raise ValueError('One of xs or ys is required')

		max_ = self.E_0
		min_ = self.get_absolute_E_max()

		if not ys:
			ys = self.get_ys(xs)

		return (ys - min_) / (max_ - min_)

	def get_x_units(self):
		return self.xs[-1].get_units()

	def get_ys(self, xs):
		return self.equation(xs, self.b, self.c, self.e)

	def __repr__(self):
		return str(self.__dict__)

def analyze_pair(model_a, model_b, model_combo):
	# print significant statistics
	ec_a = model_a.effective_concentration(model_combo.cocktail.effect / 100)
	print(f'Conc. of {model_a.cocktail} predicted to reach E{model_combo.cocktail.effect}: {ec_a}')

	intersections = model_a.get_intersection(model_b, [ec_a/8, ec_a/4, ec_a/2, ec_a, ec_a*2],
		model_combo.cocktail.ratio)
	intersections = filter_valid(intersections, minimum=1, tolerance=1)
	for intersection in intersections:
		concentration_a = intersection
		concentration_b = intersection / model_combo.cocktail.ratio
		score_a = model_a.get_ys(concentration_a)
		score_b = model_b.get_ys(concentration_b)
		print((
			f'Intersection point for {model_a.cocktail}: '
			f'({concentration_a}{model_a.get_x_units()}, {score_a})'
		))
		print((
			f'Intersection point for {model_b.cocktail}: '
			f'({concentration_b}{model_b.get_x_units()}, {score_b})'
		))
		e_experimental = 1 - model_a.get_pct_survival(ys=score_a)
		print(f'Actual effect: {(e_experimental * 100):.1f}% inhibition')

		concentration_combo_simpl = (concentration_a / 2) + (concentration_b / 2)
		score_combo_simpl = model_combo.get_ys(concentration_combo_simpl)
		concentration_combo_simpl_a = concentration_combo_simpl * \
			model_combo.cocktail.ratio.to_proportion()
		concentration_combo_simpl_b = concentration_combo_simpl * \
			model_combo.cocktail.ratio.reciprocal().to_proportion()
		print((
			f'Simplistically equivalent combo: '
			f'({concentration_combo_simpl_a}{model_a.get_x_units()} {model_a.cocktail} + '
			f'{concentration_combo_simpl_b}{model_b.get_x_units()} {model_b.cocktail}, '
			f'{score_combo_simpl})'
		))

		concentration_combo_theor = get_combo_additive_expectation(
			e_experimental, model_a, model_b, model_combo)
		score_combo_theor = model_combo.get_ys(concentration_combo_theor)
		concentration_combo_theor_a = concentration_combo_theor * \
			model_combo.cocktail.ratio.to_proportion()
		concentration_combo_theor_b = concentration_combo_theor * \
			model_combo.cocktail.ratio.reciprocal().to_proportion()
		print((
			f'Theoretically equivalent combo: '
			f'({concentration_combo_theor_a}{model_a.get_x_units()} {model_a.cocktail} + '
			f'{concentration_combo_theor_b}{model_b.get_x_units()} {model_b.cocktail}, '
			f'{score_combo_theor})'
		))

		concentration_combo_exper = model_combo.effective_concentration(e_experimental)
		score_combo_exper = model_combo.get_ys(concentration_combo_exper)
		concentration_combo_exper_a = concentration_combo_exper * \
			model_combo.cocktail.ratio.to_proportion()
		concentration_combo_exper_b = concentration_combo_exper * \
			model_combo.cocktail.ratio.reciprocal().to_proportion()
		print((
			f'Equipotent combo: '
			f'({concentration_combo_exper_a}{model_a.get_x_units()} {model_a.cocktail} + '
			f'{concentration_combo_exper_b}{model_b.get_x_units()} {model_b.cocktail}, '
			f'{score_combo_exper})'
		))

		fic = get_combo_FIC(e_experimental, model_a, model_b, model_combo)
		print(f'{model_combo.cocktail} FIC_{(e_experimental * 100):.0f}={fic:.2f}')

def chart_pair(model_a, model_b, model_combo):
	# chart A and B on the same axes, with the same x values

	model_b_scaled = Model(np.array(model_b.xs) * model_combo.cocktail.ratio, model_b.ys,
		model_b.cocktail, model_b.E_0, model_b.E_max)

	model_a.chart(close=False)
	model_b_scaled.chart(color='tab:blue', label=False,
		name=f'{model_a.cocktail}_w_adj_{model_b.cocktail}_overlay_@{model_combo.cocktail.effect}')

	# heatmap
	data = pd.DataFrame({
		'concentration': list(model_combo.xs) * 3,
		'score': np.concatenate((
			model_a.get_ys(model_a.xs),
			model_b.get_ys(model_b.xs),
			model_combo.get_ys(model_combo.xs)
		)),
		'condition': [model_a.get_condition()] * len(model_a.xs)
			+ [model_b.get_condition()] * len(model_b.xs)
			+ [model_combo.get_condition()] * len(model_combo.xs)
	})
	data = data.pivot_table(
		index='condition', columns='concentration', values='score', aggfunc='median')

	sns.heatmap(data,
		vmin=model_a.get_absolute_E_max(), vmax=model_a.E_0, cmap='viridis', annot=True, fmt='.1f',
		linewidths=1, square=True)
	plt.title(f'{model_a.get_condition()} vs. {model_b.get_condition()}: Model')
	uniq_str = str(int(time() * 1000) % 1_620_000_000_000)
	plt.savefig(
		f'{LOG_DIR}/combo_{model_a.get_condition()}-{model_b.get_condition()}_model_{uniq_str}.png')
	plt.clf()

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
# returns b_i
def do_additive_isobole(a_i, A_E50_a, B_E50_b, E_max_a, E_max_b, B_i, p, q):
	return B_i - B_E50_b/((E_max_b/E_max_a)*(1 + A_E50_a**q/a_i**q) - 1)**(1/p)

# derived from Grabovsky and Tallarida 2004, http://doi.org/10.1124/jpet.104.067264, Eq. 3
# for details see above
# returns the FIC score
def do_FIC(a_i, b_i, A_E50_a, B_E50_b, E_max_a, E_max_b, B_i, p, q):
	return (b_i + B_E50_b/((E_max_b/E_max_a)*(1 + A_E50_a**q/a_i**q) - 1)**(1/p)) / B_i

def filter_valid(array, minimum=None, tolerance=None):
	if minimum is not None:
		array = [element for element in array if element >= minimum]

	if tolerance is not None:
		blacklist = []
		for i, element in enumerate(array):
			if i not in blacklist:
				for j, element_j in enumerate(array):
					if i != j and j not in blacklist and \
							util.equalsish(element, element_j, delta=tolerance):
						blacklist.append(j)
		blacklist.sort(reverse=True)
		blacklist = dict.fromkeys(blacklist)
		for removable_idx in blacklist:
			del array[removable_idx]

	return array

def get_combo_additive_expectation(pct_inhibition, model_a, model_b, model_combo, plot=True):
	# set model_b to the model with the higher maximum effect = lower survival at maximum effect
	combo_ratio_a = model_combo.cocktail.ratio
	if model_a.c < model_b.c:
		model_a, model_b = model_b, model_a
		combo_ratio_a = combo_ratio_a.reciprocal()

	ec_a_alone = model_a.effective_concentration(pct_inhibition)
	ec_b_alone = model_b.effective_concentration(pct_inhibition)

	if np.isnan(ec_b_alone):
		return np.nan

	inhibition_max_a = 1 - model_a.get_pct_survival(ys=model_a.c)
	inhibition_max_b = 1 - model_a.get_pct_survival(ys=model_b.c)

	# find intersection between dose ratio and additive isobole
	f_isobole = lambda ec_combo_a: do_additive_isobole(
		ec_combo_a, model_a.e, model_b.e, inhibition_max_a, inhibition_max_b, ec_b_alone, model_b.b,
		model_a.b)
	f_diagonal = lambda ec_combo_a: ec_combo_a / combo_ratio_a
	simplistic_additive_estimate = 1 if np.isnan(ec_a_alone) else ec_a_alone / 2

	if plot:
		plot_func(model_a.xs, f_isobole, f'{model_combo.cocktail} Additive Isobole',
			f'{model_combo.cocktail}_isobole', f'{model_a.cocktail} ({model_a.get_x_units()})',
			close=False, color='tab:blue',
			max_x=(ec_a_alone if not np.isnan(ec_a_alone) else model_a.effective_concentration(model_a.c + 1)),
			min_y=0)
		plot_func(model_a.xs, f_diagonal, f'{model_combo.cocktail} Ratio', None, close=False,
			color='tab:green',
			max_x=(ec_a_alone if not np.isnan(ec_a_alone) else model_a.effective_concentration(model_a.c + 1)))
		plot_func(model_a.xs, lambda xs: ec_b_alone*(1 - xs/ec_a_alone),
			f'Line of simplistic additivity', f'{model_combo.cocktail}_isobole', color='lightgrey',
			linestyle='dashed', min_y=0)

	conc_b = get_intersection(f_isobole, f_diagonal, simplistic_additive_estimate)[0]
	conc_a = conc_b * combo_ratio_a
	return conc_a + conc_b

def get_combo_FIC(pct_inhibition, model_a, model_b, model_combo):
	# set model_b to the model with the higher maximum effect = lower survival at maximum effect
	combo_ratio_a = model_combo.cocktail.ratio
	if model_a.c < model_b.c:
		model_a, model_b = model_b, model_a
		combo_ratio_a = combo_ratio_a.reciprocal()

	ec_b_alone = model_b.effective_concentration(pct_inhibition)
	ec_combo = model_combo.effective_concentration(pct_inhibition)

	if np.isnan(ec_b_alone) or np.isnan(ec_combo):
		return np.nan

	ec_combo_a = ec_combo * combo_ratio_a.to_proportion()
	ec_combo_b = ec_combo * combo_ratio_a.reciprocal().to_proportion()
	inhibition_max_a = 1 - model_a.get_pct_survival(ys=model_a.c)
	inhibition_max_b = 1 - model_a.get_pct_survival(ys=model_b.c)

	return do_FIC(ec_combo_a, ec_combo_b, model_a.e, model_b.e, inhibition_max_a, inhibition_max_b,
		ec_b_alone, model_b.b, model_a.b)

def get_intersection(f1, f2, guess):
	f_intersection_equals_zero = \
		lambda xs: np.array(f1(xs), dtype=np.float64) - np.array(f2(xs), dtype=np.float64)

	try:
		with warnings.catch_warnings():
			warnings.simplefilter('ignore', RuntimeWarning)
			return scipy.optimize.root(f_intersection_equals_zero, guess, method='lm').x
	except scipy.optimize.nonlin.NoConvergence as e:
		return e.args[0]

# Ritz 2009, https://doi.org/10.1002/etc.7, Eq. 2
# `xs` is a numpy array of x values; b, c, d, and e are model parameters:
# relative slope at inflection point, lower asymptote, upper asymptote, inflection point (EC_50)
# returns y values
def log_logistic_model(xs, b, c, d, e):
	return c + (d - c) / (1 + (xs / e)**b)

def neo_E_max():
	neo_model = _get_neo_model()
	return _neo_model.get_condition_E_max()

def plot_func(xs, func, label, filename_prefix, x_label=None, close=True, color='darkgrey',
		linestyle='solid', max_x=None, max_y=None, min_x=None, min_y=None):
	if x_label:
		plt.xlabel(x_label)
	line_xs = np.linspace(0, float(max(xs)), 100)
	if max_x is not None:
		line_xs = line_xs[line_xs <= max_x]
	if min_x is not None:
		line_xs = line_xs[line_xs >= min_x]
	with warnings.catch_warnings():
		warnings.simplefilter('ignore', RuntimeWarning)
		line_ys = func(line_xs)
	if max_y is not None:
		line_xs = line_xs[line_ys <= max_y]
		line_ys = line_ys[line_ys <= max_y]
	if min_y is not None:
		line_xs = line_xs[line_ys >= min_y]
		line_ys = line_ys[line_ys >= min_y]
	plt.plot(line_xs, line_ys, color=color, label=label, marker=None)
	plt.legend()
	if close:
		uniq_str = str(int(time() * 1000) % 1_620_000_000_000)
		plt.savefig(os.path.join(LOG_DIR, f'{filename_prefix}_{uniq_str}.png'))
		plt.close()
		plt.clf()

def _get_neo_model():
	global _neo_model
	if _neo_model == None:
		xs, ys = [], []

		with open(os.path.join(util.get_here(), 'examples/neo_data.csv'),
				encoding='utf8', newline='') as f:
			for x, y in csv.reader(f, delimiter='\t'):
				xs.append(util.Solution(f'Neomycin {x}μM'))
				ys.append(float(y))

		_neo_model = Model(xs, ys, util.Cocktail('Neomycin'), debug=1)

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

	model_a = Model(model.xs, model.ys, util.Cocktail('Neo1'))
	model_b = Model(model.xs, model.ys, util.Cocktail('Neo2'))
	model_combo = Model(
		[x.dilute(0.5).combine_doses(x.dilute(0.5)) for x in model.xs], model.ys,
		util.Cocktail(('Neo1', 'Neo2'), effect=50, ratio=util.Ratio(1, 1)))

	chart_pair(model_a, model_b, model_combo)
	analyze_pair(model_a, model_b, model_combo)
