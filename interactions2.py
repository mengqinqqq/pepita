import numpy as np
import scipy.optimize
import scipy.stats

rng = np.random.default_rng()

def normalize(values, maximum=100, minimum=0):
	return (values - minimum) / (maximum - minimum)

# as per formulas in Zhao 2014, https://doi.org/10.1177/1087057114521867
def response_surface(doses_a, responses_all_a, doses_b, responses_all_b, doses_a_ab, doses_b_ab,
		responses_all_ab, positive_control, sampling_iterations=100, sample_size=10, model_size=4,
		alpha=0.05):
	positive_control_value = np.nanmean(positive_control)
	responses_all_a = normalize(responses_all_a, maximum=100, minimum=positive_control_value)
	responses_all_b = normalize(responses_all_b, maximum=100, minimum=positive_control_value)
	responses_all_ab = normalize(responses_all_ab, maximum=100, minimum=positive_control_value)

	est_true_responses_a = np.nanmean(responses_all_a, axis=1)
	est_true_responses_b = np.nanmean(responses_all_b, axis=1)

	# covariance with nans ignored
	est_response_covarmat_a = np.ma.cov(np.ma.masked_invalid(responses_all_a))
	est_response_covarmat_b = np.ma.cov(np.ma.masked_invalid(responses_all_b))

	# feels like the error/variance in responses_all_ab should be taken into account?
	observed_responses_ab = np.nanmean(responses_all_ab, axis=2)
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
		raise ValueException('Model with %d parameters is not defined' % model_size)

	gamma_sample_means = np.zeros((model_size, sampling_iterations))
	gamma_sample_covars = np.zeros((model_size, model_size, sampling_iterations))

	for sample_i in range(sampling_iterations):
		gammas = np.zeros((model_size, sample_size))
		for point_j in range(sample_size):
			noises_a = rng.multivariate_normal(np.zeros(len(doses_a)), est_response_covarmat_a)
			noises_b = rng.multivariate_normal(np.zeros(len(doses_b)), est_response_covarmat_b)

			est_theoretical_responses_ab = np.zeros((len(doses_a), len(doses_b)))

			for idx_a in range(len(doses_a)):
				for idx_b in range(len(doses_b)):
					est_theoretical_responses_ab[idx_a, idx_b] = \
						est_true_responses_a[idx_a] + est_true_responses_b[idx_b] \
						- est_true_responses_a[idx_a]*est_true_responses_b[idx_b] \
						+ noises_a[idx_a] + noises_b[idx_b] \
						- noises_a[idx_a]*noises_b[idx_b]

			est_theoretical_responses_ab = est_theoretical_responses_ab[valid_combo_idxs]

			model_j = scipy.optimize.least_squares(model_function, gamma_guess_0,
				args=(doses_a_ab, doses_b_ab, observed_responses_ab, est_theoretical_responses_ab))

			gammas[:, point_j] = model_j.x

		gamma_sample_means[:, sample_i] = np.mean(gammas, axis=1)
		gamma_sample_covars[:, :, sample_i] = np.cov(gammas)

	est_gamma = np.mean(gamma_sample_means, axis=1)
	est_gamma_covarmat = np.mean(gamma_sample_covars, axis=2) + np.cov(gamma_sample_means)

	z = scipy.stats.norm.ppf(1 - alpha/2)

	interaction_index_estimates = np.zeros((len(doses_a), len(doses_b)))
	interaction_index_ci_lowers = np.zeros((len(doses_a), len(doses_b)))
	interaction_index_ci_uppers = np.zeros((len(doses_a), len(doses_b)))

	for idx_a in range(len(doses_a)):
		dose_a = doses_a[idx_a]

		for idx_b in range(len(doses_b)):
			dose_b = doses_b[idx_b]

			x = np.array([1, dose_a, dose_b, dose_a * dose_b, dose_a**2, dose_b**2])
			x = x[:model_size] # trim to appropriate size

			interaction_index_estimate = sum(x * est_gamma)
			interaction_index_uncertainty = \
				z * np.sqrt(np.matmul(np.matmul(x, est_gamma_covarmat), x))
			interaction_index_estimates[idx_a, idx_b] = interaction_index_estimate
			interaction_index_ci_lowers[idx_a, idx_b] = \
				interaction_index_estimate - interaction_index_uncertainty
			interaction_index_ci_uppers[idx_a, idx_b] = \
				interaction_index_estimate + interaction_index_uncertainty

	results = []

	for idx_a in range(len(doses_a)):
		results.append([])
		for idx_b in range(len(doses_b)):
			results[idx_a].append('{:.3f}, ({:.3f}, {:.3f})'.format(
				interaction_index_estimates[idx_a, idx_b],
				interaction_index_ci_lowers[idx_a, idx_b],
				interaction_index_ci_uppers[idx_a, idx_b]))

	print(results)

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
		[22.5, 45, 90, 180],
		[22.5, 45, 90, 180],
		[22.5, 45, 90, 180],
		[22.5, 45, 90, 180],
	])
	doses_b_ab = np.array([
		[1.7, 1.7, 1.7, 1.7],
		[3.4, 3.4, 3.4, 3.4],
		[6.8, 6.8, 6.8, 6.8],
		[13.6, 13.6, 13.6, 13.6],
	])
	responses_all_ab = np.array([
		[
			[89.0, 115.0, 86.0, 74.0],
			[66.0, 92.0, 48.0, 59.0],
			[25.0, 46.0, 63.0, 47.0],
			[47.0, 20.0, 19.0, 24.0],
		],
		[
			[55.0, 86.0, 40.0, 50.0],
			[23.0, 64.0, 63.0, 16.0],
			[37.0, 45.0, 67.0, 34.0],
			[15.0, 20.0, 20.0, 29.0],
		],
		[
			[50.0, 41.0, 58.0, 11.0],
			[25.0, 14.0, 25.0, 27.0],
			[56.0, 50.0, 26.0, 29.0],
			[np.nan, np.nan, np.nan, np.nan],
		],
		[
			[23.0, 28.0, 38.0, 39.0],
			[28.0, 31.0, 8.0, 7.0],
			[13.0, 17.0, 42.0, 11.0],
			[np.nan, np.nan, np.nan, np.nan],
		]
	])
	positive_control = np.array([10.0, 6.0, 6.0, 5.0])

	print('Four-parameter model:')
	response_surface(
		doses_a, responses_all_a, doses_b, responses_all_b, doses_a_ab, doses_b_ab,
		responses_all_ab, positive_control, model_size=4)
	print('Six-parameter model:')
	response_surface(
		doses_a, responses_all_a, doses_b, responses_all_b, doses_a_ab, doses_b_ab,
		responses_all_ab, positive_control, model_size=6)
