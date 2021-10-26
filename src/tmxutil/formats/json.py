from tmxutil.types import Writer, TranslationUnit
from typing import TextIO
from textwrap import indent
from json import dumps


class JSONWriter(Writer):
	indent = '  '

	def __init__(self, fh: TextIO):
		self.fh = fh

	def __enter__(self) -> 'JSONWriter':
		print('[', file=self.fh)
		self.first = True
		return self

	def __exit__(self, type, value, traceback) -> None:
		if type is not None:
			return
		print(']', file=self.fh)

	def write(self, unit: TranslationUnit) -> None:
		if self.first:
			comma = ''
			self.first = False
		else:
			comma = ','
		
		print(comma + indent(dumps(unit, indent=self.indent), self.indent), file=self.fh)

