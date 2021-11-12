import csv
import numpy as np
import os.path
import pandas as pd
from matplotlib.ticker import PercentFormatter
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
		if ys and len(ys) >= 3:
			with warnings.catch_warnings():
				warnings.simplefilter('ignore', RuntimeWarning)
				warnings.simplefilter('ignore', scipy.optimize.OptimizeWarning)
				popt, pcov = scipy.optimize.curve_fit(self.equation, self.xs, self.ys)
			self.b, self.c, self.e = popt
		else:
			self.b, self.c, self.e = None, None, None

		if debug > 0:
			self.chart()

	def __repr__(self):
		return "{}({})".format(self.__class__.__name__, self.cocktail)

	def chart(self, close=True, color='darkgrey', label=True, name=None):
		plt.scatter(self.xs, self.ys, color='black', label='Data', marker='.')
		if label:
			plt.xlabel(f'{self.get_condition()} Dose (μM)')
			plt.ylabel('Pipeline Score')

		if self.b:
			line_xs = np.linspace(0, float(max(self.xs)), 100)
			line_ys = self.get_ys(line_xs)
			plt.plot(line_xs, line_ys, color=color, label='Model')
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

def analyze_checkerboard(model_a, model_b, models_combo):
	pass

def analyze_diamond(model_a, model_b, model_combo):
	# print significant statistics
	ec_a = model_a.effective_concentration(model_combo.cocktail.effect / 100)

	intersections = model_a.get_intersection(model_b, [ec_a/8, ec_a/4, ec_a/2, ec_a, ec_a*2],
		model_combo.cocktail.ratio)
	intersections = filter_valid(intersections, minimum=1, tolerance=1)

	combo_ratio_a = model_combo.cocktail.ratio
	if model_a.c < model_b.c:
		model_a, model_b = model_b, model_a
		combo_ratio_a = combo_ratio_a.reciprocal()
		intersections = [intersection * combo_ratio_a for intersection in intersections]

	fig = plt.figure()
	fig.set_size_inches(12, 8)
	fig.set_dpi(100)
	ax = fig.add_subplot(1, 1, 1)
	ax.margins(0.006)

	f_diagonal = lambda ec_combo_a: ec_combo_a / combo_ratio_a
	plot_func(model_a.xs, f_diagonal, 'Diagonal', None, close=False,
		color='lightgrey', max_x=max(model_a.xs), min_x=0,
		x_label=f'{model_a.cocktail} Dose ({model_a.get_x_units()})',
		y_label=f'{model_b.cocktail} Dose ({model_b.get_x_units()})')

	max_x, max_y = 0, 0

	for intersection in intersections:
		concentration_a = intersection
		concentration_b = intersection / combo_ratio_a
		e_experimental = 1 - model_a.get_pct_survival(xs=concentration_a)

		print(f'Intercept 1: ({concentration_a}{model_a.get_x_units()} {model_a.cocktail}, 0)')
		print(f'Intercept 2: (0, {concentration_b}{model_b.get_x_units()} {model_b.cocktail})')
		plt.scatter(concentration_a, 0, color='black',
			label=f'Equipotent Single Dose, $EC_{{{(e_experimental * 100):.0f}}}$', s=16)
		plt.scatter(0, concentration_b, color='black', s=16)

		concentration_combo_theor = get_combo_additive_expectation(
			e_experimental, model_a, model_b, model_combo, combo_ratio_a, plot=False)
		concentration_combo_theor_a = concentration_combo_theor * combo_ratio_a.to_proportion()
		concentration_combo_theor_b = concentration_combo_theor * \
			combo_ratio_a.reciprocal().to_proportion()

		plt.scatter(concentration_combo_theor_a, concentration_combo_theor_b,
			color='lightslategrey',
			label=f'Expected Equipotent Combo, $EC_{{{(e_experimental * 100):.0f}}}$', s=16)
		print((
			f'Theoretical equipotent combo: '
			f'({concentration_combo_theor_a}{model_a.get_x_units()} {model_a.cocktail}, '
			f'{concentration_combo_theor_b}{model_b.get_x_units()} {model_b.cocktail})'
		))

		concentration_combo_exper = model_combo.effective_concentration(e_experimental)
		concentration_combo_exper_a = concentration_combo_exper * combo_ratio_a.to_proportion()
		concentration_combo_exper_b = concentration_combo_exper * \
			combo_ratio_a.reciprocal().to_proportion()

		color = 'tab:red' if concentration_combo_exper > concentration_combo_theor else 'tab:green'
		plt.scatter(concentration_combo_exper_a, concentration_combo_exper_b, color=color,
			label=f'Observed Equipotent Combo, $EC_{{{(e_experimental * 100):.0f}}}$', s=16)
		print((
			f'Observed equipotent combo: '
			f'({concentration_combo_exper_a}{model_a.get_x_units()} {model_a.cocktail}, '
			f'{concentration_combo_exper_b}{model_b.get_x_units()} {model_b.cocktail})'
		))

		fic = get_combo_FIC(e_experimental, model_a, model_b, model_combo, combo_ratio_a)
		print(f'FIC_{(e_experimental * 100):.0f}={fic:.2f} for {model_combo.cocktail}')
		offset_x = max(model_a.xs) / 64
		offset_y = max(model_b.xs) / 128
		ax.annotate(f'$FIC_{{{(e_experimental * 100):.0f}}}={fic:.2f}$',
			xy=(concentration_combo_exper_a + offset_x, concentration_combo_exper_b - offset_y),
			textcoords='data')

		inhibition_max_a = 1 - model_a.get_pct_survival(ys=model_a.c)
		inhibition_max_b = 1 - model_a.get_pct_survival(ys=model_b.c)
		f_isobole = lambda ec_combo_a: do_additive_isobole(
			ec_combo_a, model_a.e, model_b.e, inhibition_max_a, inhibition_max_b, concentration_b,
			model_b.b, model_a.b)

		plot_func(model_a.xs, f_isobole,
			f'{model_combo.cocktail} $EC_{{{(e_experimental * 100):.0f}}}$', None, close=False,
			color='tab:gray',
			max_x=concentration_a, max_y=concentration_b, min_x=0, min_y=0)

		max_x = max(concentration_a + offset_x, concentration_combo_theor_a + offset_x,
			concentration_combo_exper_a + offset_x, max_x)
		max_y = max(concentration_b + offset_y, concentration_combo_theor_b + offset_y,
			concentration_combo_exper_b + offset_y, max_y)

	plt.xlim(right=max_x)
	plt.ylim(top=max_y)
	uniq_str = str(int(time() * 1000) % 1_620_000_000_000)
	plt.savefig(os.path.join(LOG_DIR, f'{model_combo.cocktail}_isoboles_{uniq_str}.png'))
	plt.close()
	plt.clf()

def chart_checkerboard(model_a, model_b, models_combo):
	label_a = f'{model_a.cocktail} Concentration ({model_a.get_x_units()})'
	label_b = f'{model_b.cocktail} Concentration ({model_b.get_x_units()})'

	data_dict = {}
	data_dict[label_a] = [float(x) for x in model_a.xs]
	data_dict[label_b] = [0] * len(model_a.xs)
	data_dict['Pct. Survival'] = [float(y) for y in model_a.get_pct_survival(ys=model_a.ys)]
	data_dict[label_a] = np.append(data_dict[label_a], [0] * len(model_b.xs))
	data_dict[label_b] = np.append(data_dict[label_b], [float(x) for x in model_b.xs])
	data_dict['Pct. Survival'] = np.append(data_dict['Pct. Survival'],
		model_b.get_pct_survival(ys=model_b.ys))

	for model_combo in models_combo:
		xs = np.array([float(x) for x in model_combo.xs])
		data_dict[label_a] = np.append(data_dict[label_a],
			xs * model_combo.cocktail.ratio.to_proportion())
		data_dict[label_b] = np.append(data_dict[label_b],
			xs * model_combo.cocktail.ratio.reciprocal().to_proportion())
		data_dict['Pct. Survival'] = np.append(data_dict['Pct. Survival'],
			model_combo.get_pct_survival(ys=model_combo.ys))

	data = pd.DataFrame(data_dict)
	data = data.pivot_table(
		index=label_a, columns=label_b, values='Pct. Survival', aggfunc='median')

	fig = plt.figure()
	fig.set_size_inches(12, 8)
	fig.set_dpi(100)
	ax = sns.heatmap(data,
		vmin=0, vmax=1, cmap='mako', annot=True, fmt='.0%', linewidths=2, square=True,
		cbar_kws={
			'format': PercentFormatter(xmax=1, decimals=0),
			'label': 'Remaining Hair-Cell Brightness', 'ticks': [0, 1]
		})
	ax.invert_yaxis()
	plt.title(f'{model_a.get_condition()} vs. {model_b.get_condition()}: Checkerboard')
	uniq_str = str(int(time() * 1000) % 1_620_000_000_000)
	plt.savefig(
		f'{LOG_DIR}/{model_a.get_condition()}-{model_b.get_condition()}_checkerboard_{uniq_str}.png'
	)
	plt.clf()

def chart_diamond(model_a, model_b, model_combo):
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

def get_combo_additive_expectation(pct_inhibition, model_a, model_b, model_combo, combo_ratio_a,
		plot=True):
	# set model_b to the model with the higher maximum effect = lower survival at maximum effect
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
			max_x=(ec_a_alone if not np.isnan(ec_a_alone) else None),
			min_y=0)
		plot_func(model_a.xs, f_diagonal, f'{model_combo.cocktail} Ratio', None, close=False,
			color='tab:green',
			max_x=(ec_a_alone if not np.isnan(ec_a_alone) else None))
		plot_func(model_a.xs, lambda xs: ec_b_alone*(1 - xs/ec_a_alone),
			f'Line of simplistic additivity', f'{model_combo.cocktail}_isobole', color='lightgrey',
			linestyle='dashed', min_y=0)

	conc_a = get_intersection(f_isobole, f_diagonal, simplistic_additive_estimate)[0]
	conc_b = conc_a / combo_ratio_a
	return conc_a + conc_b

def get_combo_FIC(pct_inhibition, model_a, model_b, model_combo, combo_ratio_a):
	# set model_b to the model with the higher maximum effect = lower survival at maximum effect
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
		linestyle='solid', max_x=None, max_y=None, min_x=None, min_y=None, y_label=None):
	if x_label:
		plt.xlabel(x_label)
	if y_label:
		plt.ylabel(y_label)
	max_x = float(max_x) if max_x is not None else float(max(xs))
	min_x = float(min_x) if min_x is not None else float(min(xs))
	line_xs = np.linspace(min_x, max_x, 100)
	if max_x is not None:
		line_xs = line_xs[line_xs <= float(max_x)]
	if min_x is not None:
		line_xs = line_xs[line_xs >= float(min_x)]
	with warnings.catch_warnings():
		warnings.simplefilter('ignore', RuntimeWarning)
		line_ys = func(line_xs)
	if max_y is not None:
		line_xs = line_xs[line_ys <= max_y]
		line_ys = line_ys[line_ys <= max_y]
	if min_y is not None:
		line_xs = line_xs[line_ys >= min_y]
		line_ys = line_ys[line_ys >= min_y]
	plt.plot(line_xs, line_ys, color=color, label=label, marker=None, zorder=-1)
	plt.legend()
	if close:
		uniq_str = str(int(time() * 1000) % 1_620_000_000_000)
		plt.savefig(os.path.join(LOG_DIR, f'{filename_prefix}_{uniq_str}.png'))
		plt.close()
		plt.clf()

def _get_model(filename, debug=1):
	xs, ys = [], []

	with open(filename, encoding='utf8', newline='') as f:
		for x, y in csv.reader(f, delimiter='\t'):
			xs.append(util.Solution(x))
			ys.append(float(y))

	return Model(xs, ys, xs[-1].get_cocktail(), debug=debug)

def _get_neo_model(debug=1):
	global _neo_model
	if _neo_model == None:
		_neo_model = _get_model(os.path.join(util.get_here(), 'examples/neo_data.csv'), debug)
	return _neo_model

#
# main
#

if __name__ == '__main__':
	if len(sys.argv) > 1:
		models = []
		for filename in sys.argv[1:]:
			models.append(_get_model(filename))
	else:
		models = [_get_neo_model()]

	for model in models:
		print(model.cocktail)
		ec_90 = model.effective_concentration(0.9)
		ec_75 = model.effective_concentration(0.75)
		ec_50 = model.effective_concentration(0.5)

		print(f'E_max: {model.get_absolute_E_max()} score')
		print(f'EC_90: {ec_90} μM')
		print(f'ec_75: {ec_75} μM')
		print(f'ec_50: {ec_50} μM')
