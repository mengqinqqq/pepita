import base64
import configparser
import csv
import hashlib
import os
import pickle
import re

_config = None
_section = 'Main'

class Dose:
	_ec_pattern = re.compile(r'([0-9]*)([A-Z]{3}[0-9]{2})/?([0-9]*)')
	_vector_pattern = re.compile(r'(.+?) ([0-9]+[.]?[0-9]*) ?([^0-9]+)')

	def __add__(self, other):
		return Dose(f'{self.drug} {self.quantity + float(other)}{self.unit}')

	def __eq__(self, other):
		if not isinstance(other, Dose):
			return False
		return self.drug == other.drug and self.quantity == other.quantity and \
			self.unit == other.unit

	def __float__(self):
		return self.quantity

	def __hash__(self):
		return hash(self.drug) ^ hash(self.quantity) ^ hash(self.unit)

	def __init__(self, string):
		self.ec = False
		self.string = string

		ec_match = Dose._ec_pattern.match(string)

		if ec_match:
			self.ec = True
			with open(os.path.join(get_here(), 'examples/ec_data.csv'),
					encoding='utf8', newline='') as f:
				ecs = {ec: vector for ec, vector in csv.reader(f, delimiter='\t')}
			multiplier, ec_data, divisor = ec_match.group(1, 2, 3)
			vector_match = Dose._vector_pattern.match(ecs.get(ec_data, None))
			self.series = ec_data
		else:
			vector_match = Dose._vector_pattern.match(string)
			self.series = None
			multiplier, divisor = None, None

		if vector_match:
			self.drug, self.quantity, self.unit = vector_match.group(1, 2, 3)
			self.quantity = float(self.quantity)
			if not self.series:
				self.series = vector_match.group(1)
		else:
			raise ValueError(f'Dose string "{string}" is not in the proper format')

		if multiplier:
			self.quantity *= int(multiplier)
		if divisor:
			self.quantity /= int(divisor)

	def __mul__(self, other):
		return Dose(f'{self.drug} {self.quantity * other}{self.unit}')

	def __radd__(self, other):
		return Dose(f'{self.drug} {float(other) + self.quantity}{self.unit}')

	def __repr__(self):
		return f'{self.drug} {self.quantity}{self.unit}'

class Ratio:
	def __eq__(self, other):
		if isinstance(other, Ratio):
			return (self.num * other.denom) == (other.num * self.denom)
		else:
			return float(self) == float(other)

	def __float__(self):
		return self.num / self.denom

	def __hash__(self):
		return hash(round(self.num / self.denom, 8))

	def __init__(self, num, denom):
		self.num = num
		self.denom = denom

	def __mul__(self, other):
		return round(other * self.num / self.denom, 8)

	def __rmul__(self, other):
		return round(other * self.num / self.denom, 8)

	def __repr__(self):
		return f'{self.num}/{self.denom}'

	def __rsub__(self, other):
		return Ratio(other*self.denom - self.num, self.denom)

	def __sub__(self, other):
		return Ratio(self.num - other*self.denom, self.denom)

	def __rtruediv__(self, other):
		return other * self.reciprocal()

	def reciprocal(self):
		return Ratio(self.denom, self.num)

class Solution:
	def __init__(self, string):
		self.string = string
		dose_strings = string.split(' + ')
		self.doses = [Dose(string) for string in dose_strings]

	def __eq__(self, other):
		if not isinstance(other, Solution):
			return False
		return self.doses == other.doses

	def __float__(self):
		return float(sum(self.doses))

	def __gt__(self, other):
		return float(self) > float(other)

	def __hash__(self):
		return hash(self.string)

	def __mul__(self, other):
		doses = [dose * other for dose in self.doses]
		return Solution(' + '.join([dose.string for dose in doses]))

	def __repr__(self):
		return '+'.join(str(dose.quantity) for dose in self.doses)

	def __rmul__(self, other):
		return other * float(self)

	def __truediv__(self, other):
		return float(self) / other

	def combine_doses(self, other):
		return Solution(f'{self.string} + {other.string}')

	def dilute(self, dilution):
		if dilution < 0 or dilution > 1:
			raise ValueError('Solution should be diluted by a factor between 0 and 1')
		doses = [dose * dilution for dose in self.doses]
		return Solution(' + '.join([dose.string for dose in doses]))

	def get_drugs(self):
		return tuple(dose.drug for dose in self.doses)

	def get_units(self):
		return '+'.join(str(dose.unit) for dose in self.doses)

	def ratio(self):
		if len(self.doses) == 2:
			return Ratio(self.doses[0].quantity, self.doses[1].quantity)
		raise ValueError(f'This solution {self.doses} does not have a valid dose ratio')

def extract_number(string):
	number_match = re.search(r'[0-9.]+', string)
	if number_match:
		return float(number_match.group(0))
	else:
		return np.nan

def get_config(setting, fallback=None):
	global _config
	if _config == None:
		_config = configparser.ConfigParser()
		_config.read(f'{get_here()}/config.ini')
		_config.read(f'{get_here()}/config-ext.ini')
	return _config[_section].get(setting, fallback)

def get_here():
	script = sys.argv[0] if __name__ == '__main__' else __file__
	return os.path.dirname(os.path.realpath(script))

def get_inputs_hashfile(**kwargs):
	sha1hash = hashlib.sha1()
	for value in kwargs.values():
		sha1hash.update(pickle.dumps(value))
	digest = base64.b32encode(sha1hash.digest()).decode('utf-8')
	return os.path.join(os.getcwd(), f'.{digest}.json')

def put_multimap(dict_, key, value):
	list_ = dict_.get(key, [])
	list_.append(value)
	dict_[key] = list_
