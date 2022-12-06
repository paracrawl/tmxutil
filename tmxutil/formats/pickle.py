import pickle
from tmxutil.types import Reader, Writer, TranslationUnit, TranslationUnitVariant
from typing import Type, Any, BinaryIO, Iterator


class TranslationUnitUnpickler(pickle.Unpickler):
	def find_class(self, module: str, name: str) -> Type[Any]:
		if module == 'tmxutil' or module == '__main__':
			if name == 'TranslationUnitVariant':
				return TranslationUnitVariant
			elif name == 'TranslationUnit':
				return TranslationUnit
		raise pickle.UnpicklingError("global '{}.{}' is forbidden".format(module, name))


class PickleReader(Reader):
	def __init__(self, fh: BinaryIO):
		self.fh = fh

	def close(self) -> None:
		self.fh.close()

	def records(self) -> Iterator[TranslationUnit]:
		try:
			while True:
				unit = TranslationUnitUnpickler(self.fh).load()
				assert isinstance(unit, TranslationUnit)
				yield unit
		except EOFError:
			pass


class PickleWriter(Writer):
	def __init__(self, fh: BinaryIO):
		self.fh = fh

	def write(self, unit: TranslationUnit) -> None:
		pickle.dump(unit, self.fh)
