from typing import TextIO
from tmxutil.types import TranslationUnit, Writer

class TxtWriter(Writer):
	def __init__(self, fh: TextIO, language: str):
		self.fh = fh
		self.language = language

	def write(self, unit: TranslationUnit) -> None:
		print(unit.translations[self.language].text, file=self.fh)

