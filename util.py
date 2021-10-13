import configparser
import os
import re

_config = None
_section = 'Main'

class Dose:
	_vector_pattern = re.compile(r'(.+?) ([0-9]+[.]?[0-9]*) ?([^0-9]+)')

	def __add__(self, other):
		return Dose(self.string.replace(str(self.quantity), str(self.quantity + other)))

	def __eq__(self, other):
		return self.drug == other.drug and self.quantity == other.quantity and \
			self.unit == other.unit

	def __float__(self):
		return self.quantity

	def __hash__(self):
		return hash(self.drug) ^ hash(self.quantity) ^ hash(self.unit)

	def __init__(self, string):
		self.string = string

		vector_match = Dose._vector_pattern.match(string)

		if vector_match:
			self.drug, self.quantity, self.unit = vector_match.group(1, 2, 3)
			self.quantity = float(self.quantity)
		else:
			raise ValueError(f'Dose string "{string}" is not in the proper format')

	def __radd__(self, other):
		return Dose(self.string.replace(str(self.quantity), str(self.quantity + other)))

	def __repr__(self):
		return f'{self.drug} {self.quantity}{self.unit}'

class Ratio:
	def __init__(self, num, denom):
		self.num = num
		self.denom = denom

	def __mul__(self, other):
		return round(other * self.num / self.denom, 5)

	def __rmul__(self, other):
		return round(other * self.num / self.denom, 5)

	def __repr__(self):
		return f'{self.num}/{self.denom}'

	def __rsub__(self, other):
		return Ratio(other*self.denom - self.num, self.denom)

	def __sub__(self, other):
		return Ratio(self.num - other*self.denom, self.denom)

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
		return float(self) * float(other)

	def __repr__(self):
		return '+'.join(str(dose.quantity) for dose in self.doses)

	def combine_doses(self, other):
		return Solution(f'{self.string} + {other.string}')

	def dilute(self, dilution):
		doses = [dose + (-dose.quantity/2) for dose in self.doses]
		return Solution(' + '.join([dose.string for dose in doses]))

	def get_drugs(self):
		return tuple(dose.drug for dose in self.doses)

	def ratio(self):
		if len(self.doses) == 2:
			return Ratio(self.doses[0].quantity, self.doses[1].quantity)
		raise ValueError(f'This solution {self.doses} does not have a valid dose ratio')

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

def put_multimap(dict_, key, value):
	list_ = dict_.get(key, [])
	list_.append(value)
	dict_[key] = list_
