import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import seaborn as sns
from time import time

import dose_response
import util

LOG_DIR = f'{util.get_config("log_dir")}/simulator'

def add_noise(n, percent=0.05, standard_dev=5):
	standard_dev = standard_dev + n * percent
	return random.gauss(n, standard_dev)

def add_static_noise(n, standard_dev=5):
	return random.gauss(n, standard_dev)

def add_percent_noise(n, coefficient_variance=0.05):
	standard_dev = n * coefficient_variance
	return random.gauss(n, standard_dev)

def main():
	simulate_noise()

def simulate_noise():
	cocktail = util.Cocktail('Test1')
	errors = {}

	for i in range(10000):
		model_real = dose_response.Model([], [], cocktail)
		model_real.b = random.uniform(0.5, 3)
		model_real.c = random.randint(6, 40)
		model_real.e = random.randint(5, 500)
		model_real.equation = lambda xs, b, c, e: dose_response.log_logistic_model(xs, b, c, 100, e)

		ec75_real = model_real.effective_concentration(0.75)

		model_real.xs = np.array([0, ec75_real/4, ec75_real/2, ec75_real, 2 * ec75_real])
		model_real.ys = model_real.get_ys(model_real.xs)

		noisy_ys = [add_noise(y) for y in model_real.ys]

		try:
			model_noisy = dose_response.Model(model_real.xs, noisy_ys, cocktail)
		except RuntimeError:
			# sometimes the random values yield an invalid result -- that's fine, move on
			continue

		ec25_real = model_real.effective_concentration(0.25)
		ec50_real = model_real.effective_concentration(0.5)
		ec90_real = model_real.effective_concentration(0.9)

		ec25_noisy = model_noisy.effective_concentration(0.25)
		ec50_noisy = model_noisy.effective_concentration(0.5)
		ec75_noisy = model_noisy.effective_concentration(0.75)
		ec90_noisy = model_noisy.effective_concentration(0.9)

		error_ec25 = abs(ec25_real - ec25_noisy) / ec25_real
		error_ec50 = abs(ec50_real - ec50_noisy) / ec50_real
		error_ec75 = abs(ec75_real - ec75_noisy) / ec75_real
		error_ec90 = abs(ec90_real - ec90_noisy) / ec90_real

		util.put_multimap(errors, 25, error_ec25)
		util.put_multimap(errors, 50, error_ec50)
		util.put_multimap(errors, 75, error_ec75)
		util.put_multimap(errors, 90, error_ec90)

	errors_df = pd.DataFrame({
		'EC value': [key for key, values in errors.items() for _ in values],
		'Percent Error': [value for key, values in errors.items() for value in values]
	})

	fig = plt.figure()
	fig.set_size_inches(12, 8)
	fig.set_dpi(100)
	sns.histplot(data=errors_df, x='Percent Error', bins=128, common_norm=False, cumulative=True,
		element='step', fill=False, hue='EC value', kde=True, log_scale=True, stat='percent')
	plt.title(f'Percent Error in the Presence of 5-score + 5% Noise')
	uniq_str = str(int(time() * 1000) % 1_620_000_000_000)
	plt.savefig(f'{LOG_DIR}/ec-noise_{uniq_str}.png')
	plt.clf()

#
# main
#

if __name__ == '__main__':
	main()
