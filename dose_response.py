import csv
from scipy.optimize import curve_fit
import warnings

# Ritz 2009, https://doi.org/10.1002/etc.7, Eq. 2
# `xs` is a numpy array of x values; b, c, d, and e are model parameters:
# relative slope at inflection point, lower asymptote, upper asymptote, inflection point (EC_50)
# returns y values
def log_logistic_model(xs, b, c, d, e):
	return c + (d - c) / (1 + (xs / e)**b)

class Model:
	def __init__(self, xs, ys, E_0=100, E_max=None):
		self.xs = xs
		self.ys = ys
		self.E_0 = E_0
		self.E_max = E_max

		model = lambda xs, b, c, e: log_logistic_model(xs, b, c, E_0, e)
		with warnings.catch_warnings():
			warnings.simplefilter("ignore", RuntimeWarning)
			popt, pcov = curve_fit(model, self.xs, self.ys)
		self.b, self.c, self.e = popt

	# pct_value = (f(x) - min) / (max - min)
	# f(x) = c + (d - c) / (1 + (x / e)**b)
	# yields x
	def find_pct(self, pct_value):
		if pct_value == 0.5:
			return self.e

		b = self.b
		c = self.c
		d = self.E_0
		e = self.e
		max_ = self.E_0
		min_ = self.get_absolute_E_max()

		return e * ((d - c) / (pct_value * (max_ - min_) + min_ - c) - 1)**(1/b)

	def get_absolute_E_max(self):
		return self.E_max if self.E_max is not None else self.c

	def get_drug_E_max(self):
		return self.c

#
# main
#

if __name__ == '__main__':
	xs, ys = [], []

	with open('examples/neo_data.csv', encoding='utf8', newline='') as f:
		for x, y in csv.reader(f, delimiter='\t'):
			xs.append(float(x))
			ys.append(float(y))

	model = Model(xs, ys)

	ic_90 = model.find_pct(0.1)
	ic_75 = model.find_pct(0.25)
	ic_50 = model.find_pct(0.5)

	print('E_max:', model.get_absolute_E_max(), 'score')
	print('IC_90:', ic_90, 'μM')
	print('IC_75:', ic_75, 'μM')
	print('IC_50:', ic_50, 'μM')
