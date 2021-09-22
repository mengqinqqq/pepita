import csv
from scipy.optimize import curve_fit
import warnings

# pct_value = (f(x) - min) / (max - min)
# f(x) = c + (d - c) / (1 + (x / e)**b)
# therefore:
# x = e * ((d - c) / (pct_value*(max - min) + min - c) - 1)**(1/b)
def find_pct(pct_value, b, c, d, e, min_, max_):
	return e * ((d - c) / (pct_value * (max_ - min_) + min_ - c) - 1)**(1/b)

# returns tuple (the parameters of the 4-param log-logistic model): b, c, d, and e:
def fit_dose_response_model(xs, ys):
	with warnings.catch_warnings():
		warnings.simplefilter("ignore", RuntimeWarning)
		popt, pcov = curve_fit(log_logistic_model, xs, ys)
	return popt

# Ritz 2009, https://doi.org/10.1002/etc.7, Eq. 2
# `xs` is a numpy array of x values; b, c, d, and e are model parameters:
# relative slope at inflection point, lower asymptote, upper asymptote, inflection point (EC_50)
# returns y values
def log_logistic_model(xs, b, c, d, e):
	return c + (d - c) / (1 + (xs / e)**b)

#
# main
#

if __name__ == '__main__':
	xs, ys = [], []

	with open('examples/neo_data.csv', encoding='utf8', newline='') as f:
		for x, y in csv.reader(f, delimiter='\t'):
			xs.append(float(x))
			ys.append(float(y))

	b, c, d, e = fit_dose_response_model(xs, ys)

	ic_90 = find_pct(0.1, b, c, d, e, c, d)
	ic_75 = find_pct(0.25, b, c, d, e, c, d)
	ic_50 = e

	print('E_max:', c, 'score')
	print('IC_90:', ic_90, 'μM')
	print('IC_75:', ic_75, 'μM')
	print('IC_50:', ic_50, 'μM')
