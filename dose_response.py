import csv
from scipy.optimize import curve_fit, OptimizeWarning
import numpy as np
import os.path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
import warnings

LOG_DIR = '/mnt/c/Users/ethan/Pictures/zebrafish/dose_response'
_neo_model = None

class Ratio:
	def __init__(self, num, denom):
		self.num = num
		self.denom = denom

	def __mul__(self, other):
		return round(other * self.num / self.denom, 5)

	def __rmul__(self, other):
		return round(other * self.num / self.denom, 5)

	def __rsub__(self, other):
		return Ratio(other*self.denom - self.num, self.denom)

	def __sub__(self, other):
		return Ratio(self.num - other*self.denom, self.denom)

class Model:
	def __init__(self, xs, ys, condition, E_0=100, E_max=None, debug=0):
		if isinstance(condition, str): # single drug
			self.combo = False
			self.proportion_a = -1
			self.xs = xs
		else: # drug combo
			self.combo = True
			self.proportion_a = Ratio(xs[-1][0], xs[-1][0] + xs[-1][1])
			self.xs = [a + b for a, b in xs]
		self.condition = condition
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

	def chart(self):
		plt.scatter(self.xs, self.ys, color='black', label='Data', marker='.')
		plt.xlabel(f'{self.get_condition()} Dose (μM)')
		plt.ylabel('Pipeline Score')

		max_x = max(self.xs)
		line_xs = np.linspace(0, max_x, 100)
		line_ys = self.get_ys(line_xs)
		plt.plot(line_xs, line_ys, color='darkgrey', label='Model')
		plt.plot(line_xs, np.ones_like(line_xs) * self.E_0, color='lightgrey', label='E_0')
		plt.plot(line_xs, np.ones_like(line_xs) * self.get_absolute_E_max(),
			color='lightgrey', label='E_max')
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
		if isinstance(self.condition, str): # single drug
			return self.condition
		else: # drug combo
			return '+'.join(self.condition)

	def get_condition_E_max(self):
		return self.c

	def get_ys(self, xs):
		return self.equation(xs, self.b, self.c, self.e)

def chart_pair(model_a, model_b, model_combo):
	# data chart

	complete_combos = []
	for combo_x, combo_y in zip(model_combo.xs, model_combo.ys):
		concentration_a = 2 * combo_x * model_combo.proportion_a
		concentration_b = 2 * combo_x * (1 - model_combo.proportion_a)
		if concentration_a in model_a.xs and concentration_b in model_b.xs:
			a_y = model_a.ys[model_a.xs.index(concentration_a)]
			b_y = model_b.ys[model_b.xs.index(concentration_b)]
			complete_combos.append((combo_x, a_y, b_y, combo_y))

	data = pd.DataFrame({
		'concentration': [combo_x for combo_x, _, _, _ in complete_combos] * 3,
		'score': [a_y for _, a_y, _, _ in complete_combos]
			+ [b_y for _, _, b_y, _ in complete_combos]
			+ [combo_y for _, _, _, combo_y in complete_combos],
		'condition': [model_a.condition] * len(complete_combos)
			+ [model_b.condition] * len(complete_combos)
			+ ['+'.join(model_combo.condition)] * len(complete_combos)
	})
	data = data.pivot_table(
		index='condition', columns='concentration', values='score', aggfunc='median')

	sns.heatmap(data,
		vmin=model_a.get_absolute_E_max(), vmax=model_a.E_0, cmap='viridis', annot=True, fmt='.1f',
		linewidths=1, square=True)
	unique_str = str(int(time() * 1000) % 1_620_000_000_000)
	plt.title(f'{model_a.condition} vs. {model_b.condition}: Raw Data')
	plt.savefig(f'{LOG_DIR}/combo_{model_a.condition}-{model_b.condition}_data_{unique_str}.png')
	plt.clf()

	# model chart

	max_x = int(max(max(model_a.xs), max(model_b.xs)))
	max_x = max_x + 5 - (max_x % 5) # round up to the nearest 5
	xs = list(range(0, int(max_x), int(max_x / 5)))
	data = pd.DataFrame({
		'concentration': xs * 3,
		'score': np.concatenate((model_a.get_ys(xs), model_b.get_ys(xs), model_combo.get_ys(xs))),
		'condition': [model_a.condition] * len(xs)
			+ [model_b.condition] * len(xs)
			+ ['+'.join(model_combo.condition)] * len(xs)
	})
	data = data.pivot_table(
		index='condition', columns='concentration', values='score', aggfunc='median')

	sns.heatmap(data,
		vmin=model_a.get_absolute_E_max(), vmax=model_a.E_0, cmap='viridis', annot=True, fmt='.1f',
		linewidths=1, square=True)
	plt.title(f'{model_a.condition} vs. {model_b.condition}: Model')
	plt.savefig(f'{LOG_DIR}/combo_{model_a.condition}-{model_b.condition}_model_{unique_str}.png')
	plt.clf()

# Berenbaum 1978, https://doi.org/10.1093/infdis/137.2.122, Eq. 1
def get_combo_FIC(pct_inhibition, model_a, model_b, model_combo, combo_proportion_a):
	ec_a = model_a.effective_concentration(pct_inhibition)
	ec_b = model_b.effective_concentration(pct_inhibition)
	ec_combo = model_combo.effective_concentration(pct_inhibition)

	if np.isnan(ec_a) or np.isnan(ec_b) or np.isnan(ec_combo):
		return np.nan

	ec_combo_a, ec_combo_b = ec_combo * combo_proportion_a, ec_combo * (1 - combo_proportion_a)
	return (ec_combo_a / ec_a) + (ec_combo_b / ec_b)

# Ritz 2009, https://doi.org/10.1002/etc.7, Eq. 2
# `xs` is a numpy array of x values; b, c, d, and e are model parameters:
# relative slope at inflection point, lower asymptote, upper asymptote, inflection point (EC_50)
# returns y values
def log_logistic_model(xs, b, c, d, e):
	return c + (d - c) / (1 + (xs / e)**b)

def neo_E_max():
	neo_model = _get_neo_model()
	return _neo_model.get_condition_E_max()

def _get_here():
	script = sys.argv[0] if __name__ == '__main__' else __file__
	return os.path.dirname(os.path.realpath(script))

def _get_neo_model():
	global _neo_model
	if _neo_model == None:
		xs, ys = [], []

		with open(os.path.join(_get_here(), 'examples/neo_data.csv'),
				encoding='utf8', newline='') as f:
			for x, y in csv.reader(f, delimiter='\t'):
				xs.append(float(x))
				ys.append(float(y))

		_neo_model = Model(xs, ys, 'Neomycin')

	return _neo_model

#
# main
#

if __name__ == '__main__':
	model = _get_neo_model()

	ec_90 = model.effective_concentration(0.9)
	ec_75 = model.effective_concentration(0.75)
	ec_50 = model.effective_concentration(0.5)

	print('E_max:', model.get_absolute_E_max(), 'score')
	print('EC_90:', ec_90, 'μM')
	print('EC_75:', ec_75, 'μM')
	print('EC_50:', ec_50, 'μM')

	chart_pair(model, model, model)
	neo_neo_FIC_50 = get_combo_FIC(0.5, model, model, model, 0.5)
	neo_neo_FIC_90 = get_combo_FIC(0.9, model, model, model, 0.5)
	print('Neomycin FIC_50, FIC_90:', neo_neo_FIC_50, neo_neo_FIC_90)
