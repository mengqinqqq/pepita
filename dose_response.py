import csv
import numpy as np
import os
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
os.makedirs(LOG_DIR, exist_ok=True)
_neo_model = None

class Model:
	def __init__(self, xs, ys, cocktail, E_0=100, E_max=None):
		self.cocktail = cocktail
		self.combo = len(cocktail.drugs) > 1
		self.xs = xs
		self.ys = ys
		self.E_0 = E_0
		self.E_max = E_max

		self.equation = lambda xs, b, c, e: log_logistic_model(xs, b, c, E_0, e)
		if ys and len(ys) >= 4:
			with warnings.catch_warnings():
				warnings.simplefilter('ignore', RuntimeWarning)
				warnings.simplefilter('ignore', scipy.optimize.OptimizeWarning)
				fits = {}
				for method in ('lm', 'trf', 'dogbox'):
					try:
						popt, _ = scipy.optimize.curve_fit(self.equation, self.xs, self.ys,
							method=method)
						sse = sum((self.equation(xs, *popt) - ys)**2)
						fits[sse] = popt
					except:
						pass
				if not fits:
					self.b, self.c, self.e = None, None, None
				else:
					self.b, self.c, self.e = fits[min(fits)]
		else:
			self.b, self.c, self.e = None, None, None

	def __repr__(self):
		return "{}({})".format(self.__class__.__name__, self.cocktail)

	def chart(self, close=True, color='darkgrey', datapoints=None, label=True, name=None,
			scale=None):
		if scale is None:
			scale = [self.E_max, self.E_0]

		ys = [(value - scale[0]) / (scale[1] - scale[0]) for value in self.ys]

		with sns.axes_style(style='darkgrid'):
			plt.scatter(self.xs, ys, color='black', marker='_', s=256)

			ax = plt.gca()
			# ax.set_ylim(bottom=0)
			ax.yaxis.set_major_formatter(PercentFormatter(1))

			if datapoints:
				scores = [value for values in datapoints.values() for value in values]
				scores = [(value - scale[0]) / (scale[1] - scale[0]) for value in scores]

				data = pd.DataFrame({
					'brightness': scores,
					'concentration': [
						float(soln) for soln, values in datapoints.items() for _ in values
					],
				})
				sns.scatterplot(
					x='concentration', y='brightness', data=data, color='black',
					label='Measurements', marker='.', s=64)

			if self.b:
				line_xs = np.linspace(0, float(max(self.xs)), 100)
				line_ys = [
					(value - scale[0]) / (scale[1] - scale[0]) for value in self.get_ys(line_xs)]
				sns.lineplot(x=line_xs, y=line_ys, label='Model')

				ec50 = self.effective_concentration(0.5)

				ec50_label = f'{ec50:.1f}{self.get_x_units()}' if not np.isnan(ec50) else 'N/A'

				plt.scatter(ec50, 0.04, color='black', label=f'EC50={ec50_label}', marker='|',
					s=128)

		if label:
			plt.title(f'{self.get_condition()} Dose-Response Curve')
			plt.xlabel(f'{self.get_condition()} Dose ({self.get_x_units()})')
			plt.ylabel('Remaining Brightness')
			plt.legend()

		if close:
			plt.tight_layout()
			uniq_str = str(int(time() * 1000) % 1_620_000_000_000)
			if not name:
				name = self.get_condition()
			plt.savefig(os.path.join(LOG_DIR, f'{name}_{uniq_str}.png'))
			plt.close()
			plt.clf()

	# pct_survival = (f(x) - min) / (max - min)
	# f(x) = c + (d - c) / (1 + (x / e)**b)
	# yields x
	def effective_concentration(self, pct_inhibition, silent=False):
		if pct_inhibition <= 0 or pct_inhibition >= 1:
			raise RuntimeError('Inhibition level must be between 0 and 1')

		if not self.b:
			return np.nan

		b = self.b
		c = self.c
		d = self.E_0
		e = self.e
		max_ = self.E_0
		min_ = self.get_absolute_E_max()
		pct_survival = 1 - pct_inhibition

		pct_pts_above_E_max = pct_survival * (max_ - min_) + min_ - c

		if pct_pts_above_E_max <= 0:
			if not silent:
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
		if xs is None and ys is None:
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

	def pivot(self):
		xs = [x.reverse() for x in self.xs]
		return Model(xs, self.ys, xs[-1].get_cocktail(), self.E_0, self.E_max)

	def __repr__(self):
		return str(self.__dict__)

def analyze_checkerboard(model_a, model_b, models_combo, method='interpolation',
		file_name_context=None, effect_level=0.5):
	file_name_context1 = ''
	file_name_context2 = ''
	if file_name_context:
		file_name_context1 = '(' + file_name_context + ')'
		file_name_context2 = file_name_context + '_'

	if method == 'interpolation':
		fig = plt.figure(figsize=(12, 8), dpi=100)
		ax = fig.add_subplot(1, 1, 1)
		ax.margins(0.006)

		swapped = False

		if model_a.c < model_b.c:
			model_a, model_b = model_b, model_a
			models_combo = [model_combo.pivot() for model_combo in models_combo]
			swapped = True

		effect_pretty = f'{(effect_level * 100):.0f}'
		ec_a = model_a.effective_concentration(effect_level)
		ec_b = model_b.effective_concentration(effect_level)
		print(f'{model_a.cocktail} EC{effect_pretty} = {ec_a:.1f}{model_a.get_x_units()}')
		print(f'{model_b.cocktail} EC{effect_pretty} = {ec_b:.1f}{model_b.get_x_units()}')

		plt.title((f'{model_a.cocktail}+{model_b.cocktail} '
			f'Predicted vs. Observed $EC_{{{effect_pretty}}}$ Isobole'))
		plt.scatter(ec_a, 0, color='black',
			label=f'Interpolated Equipotent Single Dose, $EC_{{{effect_pretty}}}$', s=16)
		plt.scatter(0, ec_b, color='black', s=16)

		fics, max_x, max_y = [], 1, 1

		for model_combo in models_combo:
			if not model_combo.b: # some combos will not have enough datapoints: skip
				continue

			ec_combo = model_combo.effective_concentration(effect_level)
			ec_combo_a = ec_combo * model_combo.cocktail.ratio.to_proportion()
			ec_combo_b = ec_combo * model_combo.cocktail.ratio.reciprocal().to_proportion()

			ec_combo_theor = get_combo_additive_expectation(
				effect_level, model_a, model_b, model_combo, model_combo.cocktail.ratio, plot=False)

			color = 'tab:red' if ec_combo > ec_combo_theor else 'tab:green'
			plt.scatter(ec_combo_a, ec_combo_b, color=color, s=16)

			fic = get_combo_FIC(effect_level, model_a, model_b, model_combo,
				model_combo.cocktail.ratio)
			fics.append(fic)

			offset_x = max(model_a.xs) / 64 # arbitrary adjustments to put text in nice location
			offset_y = max(model_b.xs) / 128
			ax.annotate(f'$FIC_{{{effect_pretty}}}={fic:.2f}$',
				xy=(ec_combo_a + offset_x, ec_combo_b - offset_y), textcoords='data')

			print((f'{model_combo.cocktail} EC{effect_pretty} = '
				f'{ec_combo_a:.1f}{model_a.get_x_units()} + {ec_combo_b:.1f}{model_b.get_x_units()}'
				f'; FIC = {fic:.2f}; ({len(model_combo.ys)} datapoints)'))

			max_x = max(ec_a + offset_x, ec_combo_a + offset_x, max_x)
			max_y = max(ec_b + offset_y, ec_combo_b + offset_y, max_y)

		ax.plot([], [], color='tab:green', fillstyle='left',
			label=f'Interpolated Equipotent Combo Doses, $EC_{{{effect_pretty}}}$',
			linestyle='none', marker='.', markerfacecoloralt='tab:red', markeredgewidth=0,
			markersize=12)

		combined_fic = util.geometric_mean(fics)

		print('Combined FIC for the whole checkerboard (by interpolation analysis):',
			f'{combined_fic:.2f}')

		inhibition_max_a = 1 - model_a.get_pct_survival(ys=model_a.c)
		inhibition_max_b = 1 - model_a.get_pct_survival(ys=model_b.c)
		f_isobole = lambda ec_combo_a: do_additive_isobole(
			ec_combo_a, model_a.e, model_b.e, inhibition_max_a, inhibition_max_b, ec_b,
			model_b.b, model_a.b)

		plot_func(model_a.xs, f_isobole,
			f'{model_a.cocktail}+{model_b.cocktail} Predicted $EC_{{{effect_pretty}}}$ Isobole',
			None, close=False, color='tab:gray', max_x=ec_a, max_y=ec_b, min_x=0, min_y=0,
			x_label=f'{model_a.cocktail} Dose ({model_a.get_x_units()})',
			y_label=f'{model_b.cocktail} Dose ({model_b.get_x_units()})')

		plt.xlim(right=max_x)
		plt.ylim(top=max_y)
		plt.tight_layout()
		uniq_str = str(int(time() * 1000) % 1_620_000_000_000)
		plt.savefig(
			os.path.join(LOG_DIR, f'{model_a.cocktail}+{model_b.cocktail}_isoboles_{uniq_str}.png'))
		plt.close()
		plt.clf()
	elif method == 'Bliss':
		# following Bansal 2014, https://doi.org/10.1038/nbt.3052

		label_a = f'{model_a.cocktail} Concentration ({model_a.get_x_units()})'
		label_b = f'{model_b.cocktail} Concentration ({model_b.get_x_units()})'

		data_dict = {}
		data_dict[label_a] = [float(x) for x in model_a.xs]
		data_dict[label_b] = [0] * len(model_a.xs)
		data_dict['Bliss Interaction'] = [np.nan for y in model_a.ys]
		data_dict[label_a] = np.append(data_dict[label_a], [0] * len(model_b.xs))
		data_dict[label_b] = np.append(data_dict[label_b], [float(x) for x in model_b.xs])
		data_dict['Bliss Interaction'] = np.append(data_dict['Bliss Interaction'],
			[np.nan for y in model_b.ys])

		for model_combo in models_combo:
			xs = np.array([float(x) for x in model_combo.xs])
			data_dict[label_a] = np.append(data_dict[label_a],
				xs * model_combo.cocktail.ratio.to_proportion())
			data_dict[label_b] = np.append(data_dict[label_b],
				xs * model_combo.cocktail.ratio.reciprocal().to_proportion())
			data_dict['Bliss Interaction'] = np.append(data_dict['Bliss Interaction'],
				[get_bliss_ixn(x, y, model_a, model_b, model_combo) \
					for x, y in zip(model_combo.xs, model_combo.ys)])

		data = pd.DataFrame(data_dict)
		data = data.pivot_table(
			index=label_a, columns=label_b, values='Bliss Interaction', aggfunc='mean',
			dropna=False)

		fig = plt.figure(figsize=(12, 8), dpi=100)
		ax = sns.heatmap(data,
			vmin=-1, vmax=1, cmap='vlag_r', center=0, annot=True, fmt='.2f', linewidths=2,
			square=True,
			cbar_kws={
				'extend': 'both',
				'label': 'Excess Over Bliss',
				# 'location': 'bottom',
				# 'shrink': 0.5,
				'ticks': [-1, 0, 1],
			})
		ax.collections[0].colorbar.set_ticklabels(
			['-1 (Antagonism)', '0 (Noninteraction)', '+1 (Synergy)'])
		ax.invert_yaxis()
		plt.title(
			f'{model_a.get_condition()} vs. {model_b.get_condition()}: Bliss Ixn ' +
				f'{file_name_context1}')
		plt.tight_layout()
		uniq_str = str(int(time() * 1000) % 1_620_000_000_000)
		plt.savefig(
			f'{LOG_DIR}/{model_a.get_condition()}-{model_b.get_condition()}_{file_name_context2}' +
				f'bliss_{uniq_str}.png'
		)
		plt.clf()
	elif method == 'Loewe':
		pass
	else:
		raise ValueException('`method` must be one of "interpolation", "Bliss", or "Loewe"')

def analyze_diamond(model_a, model_b, model_combo):
	if model_a.c is None or model_b.c is None:
		print('Cannot analyze diamond due to failed dose-response curve fit')
		return None, 0, 0

	# print significant statistics
	if model_a.c < model_b.c:
		model_a, model_b = model_b, model_a
		model_combo = model_combo.pivot()

	ax = plt.gca()

	f_diagonal = lambda ec_combo_a: ec_combo_a / model_combo.cocktail.ratio
	plot_func(model_a.xs, f_diagonal, 'Diagonal', None, close=False,
		color='lightgrey', max_x=max(model_a.xs), min_x=0,
		x_label=f'{model_a.cocktail} Dose ({model_a.get_x_units()})',
		y_label=f'{model_b.cocktail} Dose ({model_b.get_x_units()})')

	max_x, max_y = 0, 0

	e_experimental = 0.5
	concentration_a = model_a.effective_concentration(e_experimental)
	concentration_b = model_b.effective_concentration(e_experimental)

	print(f'Intercept 1: ({concentration_a}{model_a.get_x_units()} {model_a.cocktail}, 0)')
	print(f'Intercept 2: (0, {concentration_b}{model_b.get_x_units()} {model_b.cocktail})')
	plt.scatter(concentration_a, 0, color='black',
		label=f'Equipotent Single Dose, $EC_{{{(e_experimental * 100):.0f}}}$', s=16)
	plt.scatter(0, concentration_b, color='black', s=16)

	concentration_combo_theor = get_combo_additive_expectation(
		e_experimental, model_a, model_b, model_combo, model_combo.cocktail.ratio, plot=False)
	concentration_combo_theor_a = concentration_combo_theor * model_combo.cocktail.ratio.to_proportion()
	concentration_combo_theor_b = concentration_combo_theor * \
		model_combo.cocktail.ratio.reciprocal().to_proportion()

	plt.scatter(concentration_combo_theor_a, concentration_combo_theor_b,
		color='lightslategrey',
		label=f'Expected Equipotent Combo, $EC_{{{(e_experimental * 100):.0f}}}$', s=16)
	print((
		f'Theoretical equipotent combo: '
		f'({concentration_combo_theor_a}{model_a.get_x_units()} {model_a.cocktail}, '
		f'{concentration_combo_theor_b}{model_b.get_x_units()} {model_b.cocktail})'
	))

	concentration_combo_exper = model_combo.effective_concentration(e_experimental)
	concentration_combo_exper_a = concentration_combo_exper * model_combo.cocktail.ratio.to_proportion()
	concentration_combo_exper_b = concentration_combo_exper * \
		model_combo.cocktail.ratio.reciprocal().to_proportion()

	color = 'tab:red' if concentration_combo_exper > concentration_combo_theor else 'tab:green'
	plt.scatter(concentration_combo_exper_a, concentration_combo_exper_b, color=color,
		label=f'Observed Equipotent Combo, $EC_{{{(e_experimental * 100):.0f}}}$', s=16)
	print((
		f'Observed equipotent combo: '
		f'({concentration_combo_exper_a}{model_a.get_x_units()} {model_a.cocktail}, '
		f'{concentration_combo_exper_b}{model_b.get_x_units()} {model_b.cocktail})'
	))

	fic = get_combo_FIC(e_experimental, model_a, model_b, model_combo, model_combo.cocktail.ratio)
	print(f'FIC_{(e_experimental * 100):.0f}={fic:.2f} for {model_combo.cocktail}')
	offset_x = max(model_a.xs) / 64 # arbitrary adjustments to put text in nice location
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

	uniq_str = str(int(time() * 1000) % 1_620_000_000_000)
	return (os.path.join(LOG_DIR, f'{model_a.cocktail}+{model_b.cocktail}_isoboles_{uniq_str}.png'),\
		max_x, max_y)

def chart_checkerboard(model_a, model_b, models_combo, file_name_context=None):
	file_name_context1 = ''
	file_name_context2 = ''
	if file_name_context:
		file_name_context1 = '(' + file_name_context + ')'
		file_name_context2 = file_name_context + '_'

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

	fig = plt.figure(figsize=(12, 8), dpi=100)
	ax = sns.heatmap(data,
		vmin=0, vmax=1, cmap='mako', annot=True, fmt='.0%', linewidths=2, square=True,
		cbar_kws={
			'format': PercentFormatter(xmax=1, decimals=0),
			'label': 'Remaining Hair-Cell Brightness', 'ticks': [0, 1],
			# 'location': 'bottom',
			# 'shrink': 0.5,
		})
	ax.invert_yaxis()
	plt.title(f'{model_a.get_condition()} vs. {model_b.get_condition()} {file_name_context1}')
	plt.tight_layout()
	uniq_str = str(int(time() * 1000) % 1_620_000_000_000)
	plt.savefig(
		f'{LOG_DIR}/{model_a.get_condition()}-{model_b.get_condition()}_{file_name_context2}' +
			f'checkerboard_{uniq_str}.png'
	)
	plt.clf()

def chart_diamond(model_a, model_b, model_combo):
	if model_a.c is None or model_b.c is None:
		print('Cannot chart diamond due to failed dose-response curve fit')
		return

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
	plt.tight_layout()
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

def get_bliss_ixn(x, y, model_a, model_b, model_combo):
	if float(x) == 0:
		return np.nan

	# concentration_a = x * model_combo.cocktail.ratio.to_proportion()
	# concentration_b = x * model_combo.cocktail.ratio.reciprocal().to_proportion()
	x_doses = {dose.drug: dose for dose in x.doses}
	dose_a = x_doses[model_a.cocktail.drugs[0]]
	dose_b = x_doses[model_b.cocktail.drugs[0]]

	model_a_doses = [solution.doses[0] for solution in model_a.xs]
	model_b_doses = [solution.doses[0] for solution in model_b.xs]

	if dose_a in model_a_doses:
		pct_survival = (model_a.ys[model_a_doses.index(dose_a)] - model_a.get_absolute_E_max()) / \
			(model_a.E_0 - model_a.get_absolute_E_max())
		e_inhibition_a = 1 - pct_survival
	else:
		print(f'WARNING: {dose_a} not found in {model_a}. Using modeled response.')
		e_inhibition_a = 1 - model_a.get_pct_survival(xs=float(dose_a))

	if dose_b in model_b_doses:
		pct_survival = (model_b.ys[model_b_doses.index(dose_b)] - model_a.get_absolute_E_max()) / \
			(model_b.E_0 - model_b.get_absolute_E_max())
		e_inhibition_b = 1 - pct_survival
	else:
		print(f'WARNING: {dose_b} not found in {model_b}. Using modeled response.')
		e_inhibition_b = 1 - model_b.get_pct_survival(xs=float(dose_b))

	fract_inhib_theor = e_inhibition_a + e_inhibition_b - e_inhibition_a * e_inhibition_b
	fract_inhib_observed = 1 - model_combo.get_pct_survival(ys=y)

	return fract_inhib_observed - fract_inhib_theor

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

def get_combo_FIC(pct_inhibition, model_a, model_b, model_combo, combo_ratio_a, silent=False):
	# set model_b to the model with the higher maximum effect = lower survival at maximum effect
	if model_a.c < model_b.c:
		model_a, model_b = model_b, model_a
		combo_ratio_a = combo_ratio_a.reciprocal()

	ec_b_alone = model_b.effective_concentration(pct_inhibition, silent)
	ec_combo = model_combo.effective_concentration(pct_inhibition, silent)

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
		plt.tight_layout()
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

	model = Model(xs, ys, xs[-1].get_cocktail(), E_max=np.nanmin(ys))
	if debug:
		model.chart()
	return model

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
		models = [_get_neo_model(debug=0)]

	colors = ['#def', '#cdf', '#bcf', '#abf', '#9af', '#89f', '#78f']

	for i, model in enumerate(models):
		print(model.cocktail)
		ec_90 = model.effective_concentration(0.9)
		ec_75 = model.effective_concentration(0.75)
		ec_50 = model.effective_concentration(0.5)

		print(f'E_max: {model.get_absolute_E_max()} score')
		print(f'EC_90: {ec_90} μM')
		print(f'ec_75: {ec_75} μM')
		print(f'ec_50: {ec_50} μM')

		model.chart(close=False, color=colors[i])

	plt.xlabel(f'{models[0].get_condition()} Dose (μM)')
	plt.ylabel('Pipeline Score')

	uniq_str = str(int(time() * 1000) % 1_620_000_000_000)
	plt.savefig(os.path.join(LOG_DIR, f'{models[0].get_condition()}_{uniq_str}.png'))
	plt.close()
	plt.clf()
