import csv
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from time import time
import warnings

LOG_DIR = '/mnt/c/Users/ethan/Pictures/zebrafish/dose_response'

# Berenbaum 1978, https://doi.org/10.1093/infdis/137.2.122, Eq. 1
def get_combo_FIC(pct_inhibition, model_a, model_b, model_combo, combo_proportion_a):
	ec_a = model_a.effective_concentration(pct_inhibition)
	ec_b = model_b.effective_concentration(pct_inhibition)

	ec_combo = model_combo.effective_concentration(pct_inhibition)
	ec_combo_a, ec_combo_b = ec_combo * combo_proportion_a, ec_combo * (1 - combo_proportion_a)

	return (ec_combo_a / ec_a) + (ec_combo_b / ec_b)

# Ritz 2009, https://doi.org/10.1002/etc.7, Eq. 2
# `xs` is a numpy array of x values; b, c, d, and e are model parameters:
# relative slope at inflection point, lower asymptote, upper asymptote, inflection point (EC_50)
# returns y values
def log_logistic_model(xs, b, c, d, e):
	return c + (d - c) / (1 + (xs / e)**b)

class Model:
	def __init__(self, xs, ys, condition, E_0=100, E_max=None, debug=0):
		self.xs = xs
		self.ys = ys
		self.condition = condition
		self.E_0 = E_0
		self.E_max = E_max

		self.equation = lambda xs, b, c, e: log_logistic_model(xs, b, c, E_0, e)
		with warnings.catch_warnings():
			warnings.simplefilter("ignore", RuntimeWarning)
			popt, pcov = curve_fit(self.equation, self.xs, self.ys)
		self.b, self.c, self.e = popt

		if debug > 0:
			self.chart()

	def chart(self):
		plt.scatter(self.xs, self.ys, color='black', label='Data', marker='.')
		plt.xlabel(f'{self.condition} Dose (μM)')
		plt.ylabel('Pipeline Score')

		max_x = max(self.xs)
		line_xs = np.linspace(0, max_x, 100)
		line_ys = self.get_ys(line_xs)
		plt.plot(line_xs, line_ys, color='darkgrey', label='Model')
		plt.plot(line_xs, np.ones_like(line_xs) * self.E_0, color='lightgrey', label='E_0')
		plt.plot(line_xs, np.ones_like(line_xs) * self.get_absolute_E_max(),
			color='lightgrey', label='E_max')
		plt.scatter(self.e, self.get_ys(self.e), color='black', label='EC_50', marker='+')

		plt.legend()
		unique_str = str(int(time() * 1000) % 1_620_000_000_000)
		plt.savefig(f'{LOG_DIR}/{self.condition}_{unique_str}.png')

	# pct_survival = (f(x) - min) / (max - min)
	# f(x) = c + (d - c) / (1 + (x / e)**b)
	# yields x
	def effective_concentration(self, pct_inhibition):
		if pct_inhibition <= 0 or pct_inhibition >= 1:
			raise RuntimeError('Inhibition level must be between 0 and 1')

		pct_survival = 1 - pct_inhibition
		if pct_survival == 0.5:
			return self.e

		b = self.b
		c = self.c
		d = self.E_0
		e = self.e
		max_ = self.E_0
		min_ = self.get_absolute_E_max()

		return e * ((d - c) / (pct_survival * (max_ - min_) + min_ - c) - 1)**(1/b)

	def get_absolute_E_max(self):
		return self.E_max if self.E_max is not None else self.c

	def get_condition_E_max(self):
		return self.c

	def get_ys(self, xs):
		return self.equation(xs, self.b, self.c, self.e)

#
# main
#

if __name__ == '__main__':
	xs, ys = [], []

	with open('examples/neo_data.csv', encoding='utf8', newline='') as f:
		for x, y in csv.reader(f, delimiter='\t'):
			xs.append(float(x))
			ys.append(float(y))

	model = Model(xs, ys, 'Neomycin', debug=1)

	ec_90 = model.effective_concentration(0.9)
	ec_75 = model.effective_concentration(0.75)
	ec_50 = model.effective_concentration(0.5)

	print('E_max:', model.get_absolute_E_max(), 'score')
	print('EC_90:', ec_90, 'μM')
	print('EC_75:', ec_75, 'μM')
	print('EC_50:', ec_50, 'μM')
