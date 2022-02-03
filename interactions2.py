import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.stats
import seaborn as sns
from time import time
import warnings

import util

LOG_DIR = f'{util.get_config("log_dir")}/interactions2'

_rng = np.random.default_rng()

def fit_model_with_noise(model_function, gamma_guess_0, doses_a, doses_b, doses_a_ab, doses_b_ab,
		est_response_covarmat_a, est_response_covarmat_b, est_true_responses_a,
		est_true_responses_b, observed_responses_ab):
	noises_a = dict(zip(doses_a,
		_rng.multivariate_normal(np.zeros(len(doses_a)), est_response_covarmat_a)))
	noises_b = dict(zip(doses_b,
		_rng.multivariate_normal(np.zeros(len(doses_b)), est_response_covarmat_b)))

	est_true_responses_a = dict(zip(doses_a, est_true_responses_a))
	est_true_responses_b = dict(zip(doses_b, est_true_responses_b))

	est_theoretical_responses_ab = []

	for dose_a, dose_b in zip(doses_a_ab, doses_b_ab):
		# Zhao 2014, Eq. 5
		est_theoretical_responses_ab.append(
			est_true_responses_a[dose_a] + est_true_responses_b[dose_b] \
			- est_true_responses_a[dose_a]*est_true_responses_b[dose_b] \
			+ noises_a[dose_a] + noises_b[dose_b] \
			- noises_a[dose_a]*noises_b[dose_b])

	doses_a_ab = np.array([float(dose_a) for dose_a in doses_a_ab])
	doses_b_ab = np.array([float(dose_b) for dose_b in doses_b_ab])

	est_theoretical_responses_ab = np.array(est_theoretical_responses_ab)

	return scipy.optimize.least_squares(model_function, gamma_guess_0,
		args=(doses_a_ab, doses_b_ab, observed_responses_ab, est_theoretical_responses_ab))

def model_4_param(gamma, doses_a, doses_b, observed_responses_ab, theoretical_responses_ab):
	gamma_0, gamma_1, gamma_2, gamma_3 = gamma

	residuals = gamma_0 + gamma_1*doses_a + gamma_2*doses_b + gamma_3*doses_a*doses_b \
		- observed_responses_ab + theoretical_responses_ab

	return residuals

def model_6_param(gamma, doses_a, doses_b, observed_responses_ab, theoretical_responses_ab):
	gamma_0, gamma_1, gamma_2, gamma_3, gamma_4, gamma_5 = gamma

	residuals = gamma_0 + gamma_1*doses_a + gamma_2*doses_b + gamma_3*doses_a*doses_b \
		+ gamma_4*doses_a**2 + gamma_5*doses_b**2 - observed_responses_ab + theoretical_responses_ab

	return residuals

# convert values to % inhibition
def normalize(values, maximum=100, minimum=0):
	return 1 - (values - minimum) / (maximum - minimum)

def plot_heatmap(name_a, name_b, units_a, units_b, doses_a, doses_b, responses_a, responses_b,
		doses_a_ab, doses_b_ab, responses_ab, I_estimates, I_ci_his, I_ci_los, model_size,
		file_name_context=None):
	label_a = f'{name_a} Concentration ({units_a})'
	label_b = f'{name_b} Concentration ({units_a})'

	if file_name_context:
		file_name_context += '_'

	dose_response_dict = {x: y for x, y in zip(doses_a, responses_a)}
	dose_response_dict.update({x: y for x, y in zip(doses_b, responses_b)})

	data_dict = {}
	data_dict[label_a] = [float(x) for x in doses_a]
	data_dict[label_b] = [0] * len(doses_a)
	data_dict['I'] = [np.nan for y in responses_a]
	data_dict['M(I)'] = [np.nan for y in responses_a]
	data_dict['CI(I, lo)'] = [np.nan for y in responses_a]
	data_dict['CI(I, hi)'] = [np.nan for y in responses_a]

	data_dict[label_a] = np.append(data_dict[label_a], [0] * len(doses_b))
	data_dict[label_b] = np.append(data_dict[label_b], [float(x) for x in doses_b])
	data_dict['I'] = np.append(data_dict['I'], [np.nan for y in responses_b])
	data_dict['M(I)'] = np.append(data_dict['M(I)'], [np.nan for y in responses_b])
	data_dict['CI(I, lo)'] = np.append(data_dict['CI(I, lo)'], [np.nan for y in responses_b])
	data_dict['CI(I, hi)'] = np.append(data_dict['CI(I, hi)'], [np.nan for y in responses_b])

	for dose_a, dose_b, response_ab, I_est, I_ci_lower, I_ci_upper in zip(
			doses_a_ab, doses_b_ab, responses_ab, I_estimates, I_ci_los, I_ci_his):
		data_dict[label_a] = np.append(data_dict[label_a], [float(dose_a)])
		data_dict[label_b] = np.append(data_dict[label_b], [float(dose_b)])

		response_a = dose_response_dict[dose_a]
		response_b = dose_response_dict[dose_b]
		response_ab_expected = response_a + response_b - response_a*response_b

		data_dict['I'] = np.append(data_dict['I'], [response_ab - response_ab_expected])
		data_dict['M(I)'] = np.append(data_dict['M(I)'], I_est)
		data_dict['CI(I, lo)'] = np.append(data_dict['CI(I, lo)'], I_ci_lower)
		data_dict['CI(I, hi)'] = np.append(data_dict['CI(I, hi)'], I_ci_upper)

	data = pd.DataFrame(data_dict)
	data['significance'] = data.apply(
		lambda row: 'S' if row['CI(I, lo)'] * row['CI(I, hi)'] > 0 else 'NS', axis=1)
	data['label'] = data.apply(lambda row: row2label(row), axis=1)

	data_values = data.pivot_table(
		index=label_a, columns=label_b, values='I', aggfunc='mean', dropna=False)
	data_annotations = data.pivot_table(
		index=label_a, columns=label_b, values='label', aggfunc='first', dropna=False)

	fig = plt.figure()
	fig.set_size_inches(12, 8)
	fig.set_dpi(100)
	ax = sns.heatmap(data_values,
		vmin=-1, vmax=1, cmap='vlag_r', center=0, annot=data_annotations, fmt='', linewidths=2,
		square=True,
		cbar_kws={
			'extend': 'both',
			'label': 'Excess Over Bliss',
			'ticks': [-1, 0, 1],
		})
	ax.collections[0].colorbar.set_ticklabels(
		['-1 (Antagonism)', '0 (Additivity)', '+1 (Synergy)'])
	ax.invert_yaxis()
	plt.title(f'{name_a} vs. {name_b}: Bliss Ixn ({model_size}-param)')
	uniq_str = str(int(time() * 1000) % 1_620_000_000_000)
	plt.savefig(
		f'{LOG_DIR}/{name_a}-{name_b}_{file_name_context}bliss_{model_size}-param_{uniq_str}.png'
	)
	plt.clf()

def print_gamma_table(gammas, gamma_ci_his, gamma_ci_los, model_size):
	parameters = ['γ₀', 'γ₁', 'γ₂', 'γ₃', 'γ₄', 'γ₅']
	parameters = parameters[:model_size] # trim to appropriate size
	variables = ['', 'a', 'b', 'ab', 'a²', 'b²']
	variables = variables[:model_size] # trim to appropriate size

	print('Model: I =', ' + '.join([param + var for param, var in zip(parameters, variables)]))

	data = pd.DataFrame({
		'Variable': variables,
		'Parameter': parameters,
		'Value': gammas,
		'CI (high)': gamma_ci_his,
		'CI (low)': gamma_ci_los,
		'Significant': [ci_lo * ci_hi > 0 for ci_lo, ci_hi in zip(gamma_ci_los, gamma_ci_his)]
	})

	print(data.transpose())

def print_mean(doses_a, doses_b, responses_a, responses_b, doses_a_ab, doses_b_ab, responses_ab):
	dose_response_dict = {x: y for x, y in zip(doses_a, responses_a)}
	dose_response_dict.update({x: y for x, y in zip(doses_b, responses_b)})

	interaction_scores = []

	for dose_a, dose_b, response_ab in zip(doses_a_ab, doses_b_ab, responses_ab):
		response_a = dose_response_dict[dose_a]
		response_b = dose_response_dict[dose_b]
		response_ab_expected = response_a + response_b - response_a*response_b

		interaction_scores.append(response_ab - response_ab_expected)

	print('Mean interaction score:', np.nanmean(interaction_scores))

# as per formulas in Zhao 2014, https://doi.org/10.1177/1087057114521867
def response_surface(doses_a, responses_all_a, doses_b, responses_all_b, doses_a_ab, doses_b_ab,
		responses_all_ab, positive_control, sampling_iterations=1000, sample_size=20, model_size=4,
		alpha=0.05, file_name_context=None):
	positive_control_value = np.nanmean(positive_control)
	responses_all_a = normalize(responses_all_a, maximum=100, minimum=positive_control_value)
	responses_all_b = normalize(responses_all_b, maximum=100, minimum=positive_control_value)
	responses_all_ab = normalize(responses_all_ab, maximum=100, minimum=positive_control_value)

	#
	# Establish a model
	#

	est_true_responses_a = np.nanmean(responses_all_a, axis=1)
	est_true_responses_b = np.nanmean(responses_all_b, axis=1)

	# might need to replace this with Zhao 2014, Eqs. 7 and 8?
	est_response_covarmat_a = np.ma.cov(np.ma.masked_invalid(responses_all_a)) # covar ignoring NaN
	est_response_covarmat_b = np.ma.cov(np.ma.masked_invalid(responses_all_b)) # covar ignoring NaN

	with warnings.catch_warnings():
		warnings.simplefilter('ignore', RuntimeWarning)
		# feels like the error/variance in responses_all_ab should be taken into account?
		observed_responses_ab = np.nanmean(responses_all_ab, axis=1)
	valid_combo_idxs = ~np.isnan(observed_responses_ab)

	doses_a_ab = doses_a_ab[valid_combo_idxs]
	doses_b_ab = doses_b_ab[valid_combo_idxs]
	observed_responses_ab = observed_responses_ab[valid_combo_idxs]

	if model_size == 4:
		model_function = model_4_param
		gamma_guess_0 = [0.5, 0, 0, 0]
	elif model_size == 6:
		model_function = model_6_param
		gamma_guess_0 = [0.5, 0, 0, 0, 0, 0]
	else:
		raise ValueException(f'Model with {model_size} parameters is not defined')

	gamma_sample_means = np.zeros((model_size, sampling_iterations))
	gamma_sample_covars = np.zeros((model_size, model_size, sampling_iterations))

	for sample_i in range(sampling_iterations):
		gammas = np.zeros((model_size, sample_size))
		for point_j in range(sample_size):
			model_j = fit_model_with_noise(model_function, gamma_guess_0, doses_a, doses_b,
				doses_a_ab, doses_b_ab, est_response_covarmat_a, est_response_covarmat_b,
				est_true_responses_a, est_true_responses_b, observed_responses_ab)

			gammas[:, point_j] = model_j.x

		gamma_sample_means[:, sample_i] = np.mean(gammas, axis=1)
		gamma_sample_covars[:, :, sample_i] = np.cov(gammas)

	est_gamma = np.mean(gamma_sample_means, axis=1)
	est_gamma_covarmat = np.mean(gamma_sample_covars, axis=2) + np.cov(gamma_sample_means)

	#
	# Use the model (mainly `est_gamma`) to predict I and CI(I)
	#

	z = scipy.stats.norm.ppf(1 - alpha/2)

	est_gamma_stddev = np.sqrt(np.diagonal(est_gamma_covarmat))

	interaction_index_estimates = []
	interaction_index_ci_his = []
	interaction_index_ci_los = []

	for dose_a, dose_b in zip(doses_a_ab, doses_b_ab):
		dose_a, dose_b = float(dose_a), float(dose_b)
		x = np.array([1, dose_a, dose_b, dose_a * dose_b, dose_a**2, dose_b**2])
		x = x[:model_size] # trim to appropriate size

		interaction_index_estimate = np.dot(x, est_gamma)
		interaction_index_uncertainty = \
			z * np.sqrt(x @ est_gamma_covarmat @ x) # @ is matrix multiplication
		interaction_index_estimates.append(interaction_index_estimate)
		interaction_index_ci_his.append(
			interaction_index_estimate + interaction_index_uncertainty)
		interaction_index_ci_los.append(
			interaction_index_estimate - interaction_index_uncertainty)

	dose_a_0 = doses_a[0]
	dose_b_0 = doses_b[0]

	plot_heatmap(dose_a_0.drug, dose_b_0.drug, dose_a_0.unit, dose_b_0.unit, doses_a, doses_b,
		est_true_responses_a, est_true_responses_b, doses_a_ab, doses_b_ab, observed_responses_ab,
		interaction_index_estimates, interaction_index_ci_his, interaction_index_ci_los, model_size,
		file_name_context=file_name_context)
	print_gamma_table(est_gamma, est_gamma + est_gamma_stddev*z, est_gamma - est_gamma_stddev*z,
		model_size)
	print_mean(doses_a, doses_b, est_true_responses_a, est_true_responses_b, doses_a_ab, doses_b_ab,
		observed_responses_ab)

def row2label(row):
	if np.isnan(row['I']):
		return ''
	else:
		return 'I = {:.2f}\nM(I) = {:.2f}\n({:.2f}, {:.2f})\n{}'.format(
			row['I'], row['M(I)'], row['CI(I, lo)'], row['CI(I, hi)'], row['significance'])

if __name__ == '__main__':
	## example/test values

	doses_a = np.array([22.5, 45, 90, 180])
	responses_all_a = np.array([
		[109.0, 108.0, 99.0, 60.0, 108.0, 121.0, np.nan],
		[76.0, 80.0, 101.0, 81.0, 106.0, 102.0, 85.0],
		[61.0, 74.0, 57.0, 75.0, 84.0, 89.0, 55.0],
		[16.0, 32.0, 82.0, 30.0, 41.0, 11.0, 24.0],
	])
	doses_b = np.array([1.7, 3.4, 6.8, 13.6])
	responses_all_b = np.array([
		[97.0, 87.0, 112.0, 106.0, 99.0, 75.0, np.nan],
		[95.0, 62.0, 68.0, 70.0, 28.0, 68.0, 46.0],
		[25.0, 46.0, 40.0, 16.0, 25.0, 36.0, 48.0],
		[29.0, 14.0, 15.0, 22.0, 30.0, 30.0, 20.0],
	])
	doses_a_ab = np.array([
		22.5, 45, 90, 180, 22.5, 45, 90, 180, 22.5, 45, 90, 180, 22.5, 45, 90, 180
	])
	doses_b_ab = np.array([
		1.7, 1.7, 1.7, 1.7, 3.4, 3.4, 3.4, 3.4, 6.8, 6.8, 6.8, 6.8, 13.6, 13.6, 13.6, 13.6
	])
	responses_all_ab = np.array([
		[89.0, 115.0, 86.0, 74.0],
		[66.0, 92.0, 48.0, 59.0],
		[25.0, 46.0, 63.0, 47.0],
		[47.0, 20.0, 19.0, 24.0],
		[55.0, 86.0, 40.0, 50.0],
		[23.0, 64.0, 63.0, 16.0],
		[37.0, 45.0, 67.0, 34.0],
		[15.0, 20.0, 20.0, 29.0],
		[50.0, 41.0, 58.0, 11.0],
		[25.0, 14.0, 25.0, 27.0],
		[56.0, 50.0, 26.0, 29.0],
		[np.nan, np.nan, np.nan, np.nan],
		[23.0, 28.0, 38.0, 39.0],
		[28.0, 31.0, 8.0, 7.0],
		[13.0, 17.0, 42.0, 11.0],
		[np.nan, np.nan, np.nan, np.nan],
	])
	positive_control = np.array([10.0, 6.0, 6.0, 5.0])

	response_surface(
		doses_a, responses_all_a, doses_b, responses_all_b, doses_a_ab, doses_b_ab,
		responses_all_ab, positive_control, model_size=4)
	response_surface(
		doses_a, responses_all_a, doses_b, responses_all_b, doses_a_ab, doses_b_ab,
		responses_all_ab, positive_control, model_size=6)
