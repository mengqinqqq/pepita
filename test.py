import numpy as np

import dose_response
import util

def test():
	#
	# util
	#

	# util.equalsish

	assert util.equalsish(2, 2)
	assert util.equalsish(1/7, 1/7)
	assert util.equalsish(0.000001, 0.000002)
	assert not util.equalsish(0.01, 0.02)
	assert not util.equalsish(1000, -1000)

	# util.extract_number

	assert np.isnan(util.extract_number('abc'))
	assert util.extract_number('easy as 123') == 123
	assert util.extract_number('remember remember the 5th of November') == 5
	assert util.extract_number('4-score and 20 years ago') == 4
	assert util.extract_number('pi is approximately 3.14159') == 3.14159

	# util.get_inputs_hashfile

	assert util.get_inputs_hashfile(dummy1=1, dummy2='two', dummy3=3.0) == \
		util.get_inputs_hashfile(dummy1=1, dummy2='two', dummy3=3.0)
	assert util.get_inputs_hashfile(dummy1=1, dummy2='two', dummy3=3.0) != \
		util.get_inputs_hashfile(dummy1=1, dummy2='two', dummy3=4.0)

	# util.put_multimap

	dict_ = {}
	util.put_multimap(dict_, 'key', 'value')
	assert dict_ == {'key': ['value']}

	# util.Cocktail

	assert util.Cocktail('A') == util.Cocktail('A')
	assert util.Cocktail('A') != util.Cocktail('B')
	assert util.Cocktail(('A', 'B'), 50, util.Ratio(1, 1)) == \
		util.Cocktail(('A', 'B'), 50, util.Ratio(1, 1))
	assert util.Cocktail(('A', 'B'), 50, util.Ratio(1, 1)) != \
		util.Cocktail(('A', 'C'), 50, util.Ratio(1, 1))
	assert util.Cocktail(('A', 'B'), 50, util.Ratio(1, 1)) != \
		util.Cocktail(('A', 'B'), 50, util.Ratio(2, 1))

	# util.Dose

	assert util.Dose('XYZ99').drug == 'XYZ'
	assert util.Dose('XYZ99').quantity == 1
	assert util.Dose('XYZ99').unit == 'μM'
	assert util.Dose('XYZ 1μM').drug == 'XYZ'
	assert util.Dose('XYZ 1μM').quantity == 1
	assert util.Dose('XYZ 1μM').unit == 'μM'
	assert util.Dose('XYZ99') == util.Dose('XYZ 1μM')
	assert util.Dose('XYZ99') != util.Dose('XYZ 2μM')
	assert util.Dose('XYZ 1μg/mL').unit == 'μg/mL'

	assert float(util.Dose('XYZ 1μM')) == 1
	assert float(util.Dose('XYZ99')) == 1

	assert util.Dose('XYZ 1μM') + 1 == util.Dose('XYZ 2μM')
	assert 1 + util.Dose('XYZ 1μM') == util.Dose('XYZ 2μM')
	assert util.Dose('XYZ 1μM') + util.Dose('XYZ 1μM') == util.Dose('XYZ 2μM')

	assert util.Dose('XYZ 1μM') * 1 == util.Dose('XYZ 1μM')
	assert util.Dose('XYZ 1μM') * 2 == util.Dose('XYZ 2μM')
	assert util.Dose('XYZ 3μM') * 4 == util.Dose('XYZ 12μM')
	assert util.Dose('XYZ 8μM') * 0.5 == util.Dose('XYZ 4μM')

	# util.Ratio

	assert util.Ratio(1, 2) == util.Ratio(1, 2)
	assert util.Ratio(1, 2) == util.Ratio(2, 4)
	assert util.Ratio(12.5, 5) == util.Ratio(25, 10)
	assert util.Ratio(1, 2) != util.Ratio(1, 4)
	assert util.Ratio(1, 2) != util.Ratio(2, 2)
	assert util.Ratio(3.3, 10) != util.Ratio(1, 3)
	assert util.Ratio(1, 2) == 0.5
	assert util.Ratio(3, 1) == 3

	assert util.equalsish(float(util.Ratio(5, 20)), 0.25)
	assert util.equalsish(float(util.Ratio(6, 10)), 0.6)
	assert util.equalsish(float(util.Ratio(11, 5)), 2.2)

	assert util.Ratio(1, 4) * util.Ratio(4, 1) == 1
	assert util.Ratio(1, 4) * util.Ratio(4, 1) == util.Ratio(1, 1)
	assert 3 * util.Ratio(4, 1) == 12
	assert 2 * util.Ratio(3, 2) == 3
	assert util.Ratio(4, 1) * 3 == 12
	assert util.Ratio(3, 2) * 2 == 3
	assert util.equalsish(util.Ratio(12.5, 5) * 5, 12.5)
	assert util.equalsish(5 * util.Ratio(12.5, 5), 12.5)

	assert util.Ratio(1, 2).reciprocal() == util.Ratio(2, 1)
	assert util.Ratio(7, 3).reciprocal() == util.Ratio(3, 7)

	assert 2 / util.Ratio(1, 2) == 4
	assert 5 / util.Ratio(5, 2) == 2
	assert 4 / util.Ratio(8, 1) == 0.5

	assert util.Ratio(1, 9).to_proportion() == util.Ratio(1, 10)
	assert util.Ratio(5, 4).to_proportion() == util.Ratio(5, 9)

	assert util.Dose('XYZ 1μM') * util.Ratio(7, 2) == util.Dose('XYZ 3.5μM')
	assert util.Dose('XYZ 6μM') * util.Ratio(2, 3) == util.Dose('XYZ 4μM')
	assert util.Dose('XYZ 32μM') / util.Ratio(2, 3) == util.Dose('XYZ 48μM')
	assert util.Dose('XYZ 12μM') / util.Ratio(4, 1) == util.Dose('XYZ 3μM')

	# util.Solution

	assert util.Solution('XYZ 1μM') == util.Solution('XYZ 1μM')
	assert util.Solution('ABC 10μg/mL') == util.Solution('ABC 10μg/mL')
	assert util.Solution('XYZ 1μM') != util.Solution('XYZ 1μg/mL')
	assert util.Solution('XYZ 1μM') != util.Solution('XYZ 10μM')
	assert util.Solution('XYZ 1μM') != util.Solution('ABC 1μM')
	assert util.Solution('XYZ99') == util.Solution('XYZ 1μM')
	assert util.Solution('XYZ 1μM + ABC 10μg/mL') == util.Solution('XYZ 1μM + ABC 10μg/mL')
	assert util.Solution('XYZ 1μM + ABC 10μg/mL') == util.Solution('XYZ99 + ABC 10μg/mL')
	assert util.Solution('XYZ 1μM').doses[0] == util.Dose('XYZ 1μM')
	assert util.Solution('XYZ 1μM + ABC 10μg/mL').doses == \
		[util.Dose('XYZ 1μM'), util.Dose('ABC 10μg/mL')]

	assert float(util.Solution('XYZ 1μM + ABC 10μg/mL')) == 11

	assert util.Solution('XYZ 10μM') > util.Solution('XYZ 2μM')

	assert util.Solution('XYZ 1μM') * 2 == util.Solution('XYZ 2μM')
	assert util.Solution('XYZ 10μM') * 0.5 == util.Solution('XYZ 5μM')

	assert 2 * util.Solution('XYZ 1μM') == 2
	assert 3 * util.Solution('XYZ 1μM + ABC 10μg/mL') == 33

	assert util.Solution('XYZ 10μM') / 2 == 5
	assert util.Solution('XYZ 32μM') / 4 == 8

	assert util.Solution('XYZ 1μM').combine_doses(util.Solution('ABC 10μg/mL')) == \
		util.Solution('XYZ 1μM + ABC 10μg/mL')
	assert util.Solution('XYZ 1μM').combine_doses(util.Solution('ABC 10μg/mL')).doses == \
		[util.Dose('XYZ 1μM'), util.Dose('ABC 10μg/mL')]

	assert util.Solution('XYZ 10μM').dilute(0.5) == util.Solution('XYZ 5μM')
	assert util.Solution('XYZ 20μM').dilute(0.2) == util.Solution('XYZ 4μM')

	assert util.Solution('XYZ 10μM').get_cocktail() == util.Cocktail('XYZ')
	assert util.Solution('XYZ 1μM + ABC 10μg/mL').get_cocktail() == \
		util.Cocktail(('XYZ', 'ABC'), ratio=util.Ratio(1, 10))

	#
	# dose_response
	#

	# dose_response.do_additive_isobole

	# values from Grabovsky and Tallarida 2004, http://doi.org/10.1124/jpet.104.067264, p. 983
	# except for return value, which has been calculated separately
	assert util.equalsish(0.46255,
		dose_response.do_additive_isobole(
			a_i=25, A_E50_a=65.8, B_E50_b=3.99, E_max_a=1.58, E_max_b=4.17, B_i=1.2, p=1.73,
			q=1.92))
	assert util.equalsish(2.06255,
		dose_response.do_additive_isobole(
			a_i=25, A_E50_a=65.8, B_E50_b=3.99, E_max_a=1.58, E_max_b=4.17, B_i=2.8, p=1.73,
			q=1.92))
	assert util.equalsish(4.46255,
		dose_response.do_additive_isobole(
			a_i=25, A_E50_a=65.8, B_E50_b=3.99, E_max_a=1.58, E_max_b=4.17, B_i=5.2, p=1.73,
			q=1.92))
	assert util.equalsish(0.60894,
		dose_response.do_additive_isobole(
			a_i=100, A_E50_a=65.8, B_E50_b=3.99, E_max_a=1.58, E_max_b=4.17, B_i=2.8, p=1.73,
			q=1.92))
	assert util.equalsish(3.00894,
		dose_response.do_additive_isobole(
			a_i=100, A_E50_a=65.8, B_E50_b=3.99, E_max_a=1.58, E_max_b=4.17, B_i=5.2, p=1.73,
			q=1.92))
	assert util.equalsish(2.28538,
		dose_response.do_additive_isobole(
			a_i=400, A_E50_a=65.8, B_E50_b=3.99, E_max_a=1.58, E_max_b=4.17, B_i=5.2, p=1.73,
			q=1.92))

	# dose_response.do_FIC

	# values as above, with FIC=1 due to the given isoboles being additive
	assert util.equalsish(1,
		dose_response.do_FIC(a_i=25, b_i=0.46255, A_E50_a=65.8, B_E50_b=3.99, E_max_a=1.58, E_max_b=4.17, B_i=1.2,
			p=1.73,q=1.92))
	assert util.equalsish(1,
		dose_response.do_FIC(a_i=25, b_i=2.06255, A_E50_a=65.8, B_E50_b=3.99, E_max_a=1.58, E_max_b=4.17, B_i=2.8,
			p=1.73,q=1.92))
	assert util.equalsish(1,
		dose_response.do_FIC(a_i=25, b_i=4.46255, A_E50_a=65.8, B_E50_b=3.99, E_max_a=1.58, E_max_b=4.17, B_i=5.2,
			p=1.73,q=1.92))
	assert util.equalsish(1,
		dose_response.do_FIC(a_i=100, b_i=0.60894, A_E50_a=65.8, B_E50_b=3.99, E_max_a=1.58, E_max_b=4.17,
			B_i=2.8, p=1.73,q=1.92))
	assert util.equalsish(1,
		dose_response.do_FIC(a_i=100, b_i=3.00894, A_E50_a=65.8, B_E50_b=3.99, E_max_a=1.58, E_max_b=4.17,
			B_i=5.2, p=1.73,q=1.92))
	assert util.equalsish(1,
		dose_response.do_FIC(a_i=400, b_i=2.28538, A_E50_a=65.8, B_E50_b=3.99, E_max_a=1.58, E_max_b=4.17,
			B_i=5.2, p=1.73,q=1.92))

	# dose_response.filter_valid

	assert dose_response.filter_valid([1, 1, 2, 3, 5, 8], minimum=3) == [3, 5, 8]
	assert dose_response.filter_valid([1, 1, 2, 3, 5, 8], tolerance=2) == [1, 3, 5, 8]
	assert dose_response.filter_valid([1, 1, 2, 3, 5, 8], tolerance=3) == [1, 5, 8]
	assert dose_response.filter_valid([1, 1, 2, 3, 5, 8], minimum=3, tolerance=3) == [3, 8]

	# dose_response.Model

	model = dose_response._get_neo_model()

	assert model.effective_concentration(0.001) < 1
	assert model.get_absolute_E_max() < 50
	assert model.get_condition_E_max() < 50
	assert util.equalsish(1, model.get_pct_survival(xs=0.001))
	assert util.equalsish(0, model.get_pct_survival(xs=2000))
	assert model.get_pct_survival(ys=100) == 1
	assert model.get_pct_survival(ys=0.001) <= 0
	assert util.equalsish(0, model.get_pct_survival(ys=model.get_absolute_E_max()))
	assert model.get_x_units() == 'μM'
	assert model.get_ys(0.001) > 99
	assert model.get_ys(2000) < 50

	# get_combo_additive_expectation(pct_inhibition, model_a, model_b, model_combo, plot=True)

#
# main
#

if __name__ == '__main__':
	test()
