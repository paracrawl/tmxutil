import csv
from tmxutil.types import Reader, Writer, TranslationUnit, TranslationUnitVariant
from typing import List, Callable, TextIO, Iterable, Any, Iterator


class TabReader(Reader):
	def __init__(self, fh: TextIO, src_lang: str, trg_lang: str, columns: Iterable[str] = ['source-document-1', 'source-document-2', 'text-1', 'text-2', 'score-aligner']):
		self.fh = fh
		self.src_lang = src_lang
		self.trg_lang = trg_lang
		self.columns = columns

	def close(self) -> None:
		self.fh.close()

	def records(self) -> Iterator[TranslationUnit]:
		class Variant:
			__slots__ = ('lang', 'unit')
			def __init__(self, lang: str):
				self.lang = lang
				self.unit = TranslationUnitVariant()

		for n, line in enumerate(self.fh):
			# Skip blank lines
			if line.strip() == '':
				continue

			values = line.rstrip('\n').split('\t')

			record = TranslationUnit(id={str(n)})

			var1 = Variant(self.src_lang)

			var2 = Variant(self.trg_lang)

			for column, value in zip(self.columns, values):
				if column == '-':
					continue

				if column.endswith('-1') or column.endswith('-2'):
					variant = var1 if column.endswith('-1') else var2

					if column[:-2] == 'lang':
						variant.lang = value
					elif column[:-2] == 'text':
						variant.unit.text = value
					else:
						variant.unit[column[:-2]] = {value}
				else:
					record.setdefault(column, set()).add(value)

			record.translations = {
				var1.lang: var1.unit,
				var2.lang: var2.unit
			}

			yield record


class TabWriter(Writer):
	fh: TextIO
	columns: List[Callable[[TranslationUnit], Iterable[Any]]]

	def __init__(self, fh: TextIO, columns: List[Callable[[TranslationUnit], Iterable[Any]]]):
		self.fh = fh
		self.columns = columns

	def __enter__(self) -> 'TabWriter':
		self.writer = csv.writer(self.fh, delimiter='\t')
		return self

	def write(self, unit: TranslationUnit) -> None:
		self.writer.writerow([
			';'.join(map(str, getter(unit)))
			for getter in self.columns
		])