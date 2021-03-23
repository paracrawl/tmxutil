#!/usr/bin/env python3
# Tool to convert between tab, txt and tmx formatting, filtering & adding
# annotations.

__VERSION__ = 1.1

import csv
import sys
import re
import gzip
import pickle
import resource
import operator
from abc import ABC, ABCMeta, abstractmethod
from argparse import ArgumentParser, FileType, Namespace
from collections import defaultdict, OrderedDict
from contextlib import contextmanager
from datetime import datetime
from functools import partial
from io import BufferedReader, TextIOWrapper
from itertools import combinations, chain, starmap
from logging import info, warning, getLogger, INFO
from pprint import pprint
from tempfile import TemporaryFile
from typing import Callable, Dict, List, Optional, Any, Iterator, Iterable, Set, Tuple, Type, TypeVar, BinaryIO, TextIO, IO, Union, cast, Generator
from types import TracebackType
from xml.sax.saxutils import escape, quoteattr
from xml.etree.ElementTree import iterparse, Element

# Only Python 3.7+ has fromisoformat
if hasattr(datetime, 'fromisoformat'):
	fromisoformat = datetime.fromisoformat
else:
	def fromisoformat(date_string: str) -> datetime:
		match = re.match(r'^(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})(?:.(?P<hour>\d{2})(?::(?P<minute>\d{2})(?::(?P<second>\d{2}))?)?)?$', date_string)
		if match is None:
			raise ValueError("invalid fromisoformat value: '{}'".format(date_string))
		return datetime(
			int(match['year']), int(match['month']), int(match['day']),
			int(match['hour']), int(match['minute']), int(match['second']))


def fromfilesize(size_string: str) -> int:
	order = 1
	for suffix in ['B', 'K', 'M', 'G', 'T']:
		if size_string.endswith(suffix):
			return int(size_string[:-1]) * order
		else:
			order *= 1000
	return int(size_string)


class TranslationUnitVariant(Dict[str, Set[str]]):
	__slots__ = ['text']

	def __init__(self, *, text: Optional[str] = None, **kwargs: Set[str]):
		super().__init__(**kwargs)
		self.text = text or ''

	def updateVariant(self, other: 'TranslationUnitVariant') -> None:
		self.text = other.text
		for key, value in other.items():
			if key in self:
				self[key] |= value
			else:
				self[key] = value


class TranslationUnit(Dict[str,Union[float, str, Set[str]]]):
	__slots__ = ['translations']

	def __init__(self, *, translations: Optional[Dict[str, TranslationUnitVariant]] = None, **kwargs: Union[float, str, Set[str]]):
		super().__init__(**kwargs)
		self.translations = translations or dict() # type: Dict[str,TranslationUnitVariant]


class BufferedBinaryIO(BinaryIO, metaclass=ABCMeta):
	@abstractmethod
	def peek(self, size: int) -> bytes:
		...


class XMLWriter(object):
	"""Writes XML. Light wrapper around iobase.write really, but handles
	properly indenting and closing xml elements at the right time."""

	def __init__(self, fh: TextIO):
		self.fh = fh
		self.stack = [] # type: List[Tuple[str,bool]]
		self.indent = '  '

	def open(self, name: str, attributes: Dict[str,Any] = dict()) -> None:
		"""Write open tag."""

		if self.stack:
			self.stack[-1] = (self.stack[-1][0], True)

		out = '\n' + (self.indent * len(self.stack)) + '<' + name

		for attr_name, attr_value in attributes.items():
			out += ' ' + attr_name + '=' + quoteattr(str(attr_value))

		out += '>'

		self.fh.write(out)

		self.stack.append((name, False))

	def close(self) -> None:
		"""Write close tag. Will use stack to determine which element."""

		name, has_children = self.stack.pop()
		if has_children:
			self.fh.write('\n{}</{}>'.format(self.indent * len(self.stack), name))
		else:
			self.fh.write('</{}>'.format(name))

	def write(self, text: Any) -> None:
		self.fh.write(escape(str(text).rstrip()))

	def element(self, name: str, attributes: Dict[str,Any] = dict(), text: str = None) -> None:
		# Notify the parent element it has children (for formatting)
		if self.stack:
			self.stack[-1] = (self.stack[-1][0], True)

		self.fh.write('\n{indent}<{name}{attr}>{text}</{name}>'.format(
			indent=self.indent * len(self.stack),
			name=name,
			text=escape(str(text).rstrip()) if text is not None else '',
			attr=''.join(
				' {}={}'.format(attr_name, quoteattr(str(attr_value)))
				for attr_name, attr_value in attributes.items()
			)))

	def __enter__(self) -> 'XMLWriter':
		self.fh.write('<?xml version=\"1.0\"?>')
		return self

	def __exit__(self, type: Optional[Type[BaseException]], value: Optional[BaseException], traceback: Optional[TracebackType]) -> None:
		if type is None:
			while len(self.stack):
				self.close()

# unit = {
# 	'score-aligner': float,
#   'score-bifixer': float,
#   'score-bicleaner': float,
#   'hash-bifixer': str,
# 	'translations': {
# 		str: {
# 			'source-document': {str},
#			'ipc': {str}
#   		'ipc-group': {str},
# 			'text': str
# 		}
# 	}
# }

class Reader(ABC):
	"""Interface for sentence pair input stream."""

	def __iter__(self) -> Iterator[TranslationUnit]:
		return self.records()

	@abstractmethod
	def records(self) -> Iterator[TranslationUnit]:
		pass


class Writer(ABC):
	"""Interface for sentence pair output stream. Has with statement context
	magic functions that can be overwritten to deal with writing headers and
	footers, or starting and ending XML output."""

	def __enter__(self) -> 'Writer':
		return self

	def __exit__(self, type: Optional[Type[BaseException]], value: Optional[BaseException], traceback: Optional[TracebackType]) -> None:
		pass

	@abstractmethod
	def write(self, unit: TranslationUnit) -> None:
		pass


class TMXReader(Reader):
	"""TMX File format reader. XML attributes are mostly ignored. <prop/>
	elements of the <tu/> are added as attributes, and of <tuv/> as attributes
	with sets of values as we expect one or more of them, i.e. one or more
	source-document, ipc, etc."""

	def __init__(self, fh: TextIO):
		self.fh = fh

	def records(self) -> Iterator[TranslationUnit]:
		path = [] # type: List[str]
		stack = [] # type: List[Element]

		info("TMXReader starts reading from %s", self.fh.name)
		
		def enter(element: Element) -> None:
			stack.append(element)
			path.append(element.tag)

		def exit(element: Element) -> None:
			removed = stack.pop()
			assert removed == element
			path.pop()
			if stack:
				# Remove element from parent to keep the internal tree empty
				stack[-1].remove(removed)

		unit = TranslationUnit() # type: TranslationUnit
		translation = TranslationUnitVariant() # type: TranslationUnitVariant

		lang_key = '{http://www.w3.org/XML/1998/namespace}lang'
		
		for event, element in iterparse(self.fh, events=('start', 'end')):
			if event == 'start':
				enter(element)

				if path == ['tmx', 'body', 'tu']:
					unit = TranslationUnit(id=element.get('tuid'))
				elif path == ['tmx', 'body', 'tu', 'tuv']:
					translation = TranslationUnitVariant()
			elif event == 'end':
				if path == ['tmx', 'body', 'tu']:
					yield unit
				elif path == ['tmx', 'body', 'tu', 'prop']:
					if element.text is None:
						warning('empty <prop type="%s"></prop> encountered in unit with id %s in file %s; property ignored', element.get('type'), unit['id'], self.fh.name)
					else:
						unit[element.get('type')] = float(element.text.strip()) if 'score' in element.get('type') else element.text.strip()
				elif path == ['tmx', 'body', 'tu', 'tuv']:
					unit.translations[element.attrib[lang_key]] = translation
					translations = None
				elif path == ['tmx', 'body', 'tu', 'tuv', 'prop']:
					if element.text is None:
						warning('empty <prop type="%s"></prop> encountered in unit with id %s in file %s; property ignored', element.get('type'), unit['id'], self.fh.name)
					else:
						if element.get('type') in translation:
							translation[element.get('type')].add(element.text.strip())
						else:
							translation[element.get('type')] = {element.text.strip()}
				elif path == ['tmx', 'body', 'tu', 'tuv', 'seg']:
					if element.text is None:
						warning('empty translation segment encountered in unit with id %s in file %s', unit['id'], self.fh.name)
						translation.text = ''
					else:
						translation.text = element.text.strip()

				exit(element)


class TMXWriter(Writer):
	def __init__(self, fh: TextIO, *, creation_date: Optional[datetime] = None):
		self.fh = fh
		self.creation_date = creation_date
		
	def __enter__(self) -> 'TMXWriter':
		self.writer = XMLWriter(self.fh)
		self.writer.__enter__()
		self.writer.open('tmx', {'version': 1.4})

		args = {
			'o-tmf': 'PlainText',
			'creationtool': 'tab2tmx.py',
			'creationtoolversion': __VERSION__,
			'datatype': 'PlainText',
			'segtype': 'sentence',
			'o-encoding': 'utf-8',
		}

		if self.creation_date is not None:
			args['creationdate'] = self.creation_date.strftime("%Y%m%dT%H%M%S")

		self.writer.element('header', args) # <header/>
		
		self.writer.open('body')
		return self

	def __exit__(self, type: Optional[Type[BaseException]], value: Optional[BaseException], traceback: Optional[TracebackType]) -> None:
		self.writer.__exit__(type, value, traceback)

	def _write_prop(self, name: str, value: Union[str,float,Set[str]]) -> None:
		if value is None:
			return
		elif isinstance(value, set):
			for val in sorted(value):
				self._write_prop(name, val)
		else:
			self.writer.element('prop', {'type': name}, value)

	def write(self, unit: TranslationUnit) -> None:
		self.writer.open('tu', {'tuid': unit['id'], 'datatype': 'Text'}) # <tu>
		for key, value in sorted(unit.items()):
			if key != 'id':
				self._write_prop(key, value) # <prop/>
		for lang, translation in sorted(unit.translations.items()):
			self.writer.open('tuv', {'xml:lang': lang}) # <tuv>
			for key, value in sorted(translation.items()):
				self._write_prop(key, value) # <prop/>
			self.writer.element('seg', text=translation.text) # <seg>..</seg>
			self.writer.close() # </tuv>
		self.writer.close() # </tu>


class TabReader(Reader):
	def __init__(self, fh: TextIO, src_lang: str, trg_lang: str, columns: List[str] = ['source-document-1', 'source-document-2', 'text-1', 'text-2', 'score-aligner']):
		self.fh = fh
		self.src_lang = src_lang
		self.trg_lang = trg_lang
		self.columns = columns

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

			values = line.split('\t')

			record = TranslationUnit(id=str(n))

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
					record[column] = value

			record.translations = {
				var1.lang: var1.unit,
				var2.lang: var2.unit
			}

			yield record


A = TypeVar('A')
B = TypeVar('B')
def first(it: Iterable[A], default: Optional[B] = None) -> Optional[Union[A,B]]:
	return next(iter(it), default)


class TabWriter(Writer):
	def __init__(self, fh: TextIO, languages: List[str] = []):
		self.fh = fh
		self.languages = languages

	def __enter__(self) -> 'TabWriter':
		self.writer = csv.writer(self.fh, delimiter='\t')
		return self

	def write(self, unit: TranslationUnit) -> None:
		if not self.languages:
			self.languages = list(unit.translations.keys())

		self.writer.writerow(
			  [first(unit.translations[lang]['source-document'], '') for lang in self.languages]
			+ [unit.translations[lang].text for lang in self.languages])


class TxtWriter(Writer):
	def __init__(self, fh: TextIO, language: str):
		self.fh = fh
		self.language = language

	def write(self, unit: TranslationUnit) -> None:
		print(unit.translations[self.language].text, file=self.fh)


class PyWriter(Writer):
	def __init__(self, fh: TextIO):
		self.fh = fh

	def write(self, unit: TranslationUnit) -> None:
		pprint(unit, stream=self.fh)


class IPCLabeler(object):
	"""Add IPC labels to sentence pairs based on the patent ids found in the
	source-document property of either side of the pair."""

	#lut: Dict[Tuple[str,str], Set[str]]

	def __init__(self, files: List[TextIO] = []):
		self.lut = dict() # type: Dict[Tuple[str,str], Set[str]]
		for fh in files:
			self.load(fh)

	def load(self, fh: TextIO) -> None:
		for line in fh:
			src_id, _, _, _, src_lang, src_ipcs, trg_id, _, _, _, trg_lang, trg_ipcs = line.split('\t', 11)
			self.lut[(src_lang.lower(), src_id)] = set(ipc.strip() for ipc in src_ipcs.split(',') if ipc.strip() != '')
			self.lut[(trg_lang.lower(), trg_id)] = set(ipc.strip() for ipc in trg_ipcs.split(',') if ipc.strip() != '')

	def annotate(self, unit: TranslationUnit) -> TranslationUnit:
		for lang, translation in unit.translations.items():
			# Ignoring type because https://github.com/python/mypy/issues/2013
			translation['ipc'] = set().union(*(
				self.lut[(lang.lower(), url)]
				for url in translation['source-document']
				if (lang.lower(), url) in self.lut
			)) # type: ignore
		return unit


class IPCGroupLabeler(object):
	"""Add overall IPC group ids based on IPC labels added by IPCLabeler."""

	#patterns: List[Tuple[str,Set[str]]]

	def __init__(self, files: List[TextIO] = []):
		self.patterns = [] # type: List[Tuple[str,Set[str]]]
		for fh in files:
			self.load(fh)

	def load(self, fh: TextIO) -> None:
		for line in fh:
			prefix, group, *_ = line.split('\t', 2)
			self.patterns.append((
				prefix.strip(),
				{prefix.strip(), group.strip()} if prefix.strip() != "" else {group.strip()}
			))

		# Sort with most specific on top
		self.patterns.sort(key=lambda pattern: (-len(pattern[0]), pattern[0]))

	def find_group(self, ipc_code: str) -> Set[str]:
		for prefix, groups in self.patterns:
			if ipc_code.startswith(prefix):
				return groups
		return set()

	def annotate(self, unit: TranslationUnit) -> TranslationUnit:
		for lang, translation in unit.translations.items():
			translation['ipc-group'] = set().union(*map(self.find_group, translation['ipc'])) # type: ignore
		return unit


def text_key(unit: TranslationUnit) -> Tuple[str,...]:
	return tuple(translation.text for translation in unit.translations.values())


def deduplicate(reader: Iterator[TranslationUnit], key: Callable[[TranslationUnit], Any], sort_key: Callable[[TranslationUnit], Any] = lambda unit: 0, mem_limit:int = 2 * 10**9) -> Iterator[TranslationUnit]:
	"""
	Deduplicate records read from reader. It does this by creating a hash table
	of all records, grouped by key(record). If multiple records have the same
	key they are combined if properties allow this (i.e. sets, lists) or
	overwritten in case compare(current, new) is True. See deduplicate_merge().
	
	Note: This function behaves like an iterator but will only start yielding
	results once reader has run out of records.

	Note: If the memory usage becomes too large (because storing all unique
	units is taking up too much storage) it will fall back to deduplicate_external
	which uses a file as backing for temporarily storing translation units.
	"""

	best = dict() # type: Dict[int,TranslationUnit]

	try:
		first_unit = next(reader)
	except StopIteration:
		return reader

	for n, unit in enumerate(chain([first_unit], reader), start=1):
		unit_id = hash(key(unit))

		if unit_id in best:
			best[unit_id] = deduplicate_merge(best[unit_id], unit, sort_key)
		else:
			best[unit_id] = unit
		
			if n % 10000 == 0:
				mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
				info('best contains %d (%d processed) entries (%1.2f GB)', len(best), n, mem_usage / 10**9)
				if mem_usage > mem_limit:
					info("Exceeded in-memory size limit, switching to file-backed deduplication")
					already_processed = best.values()
					del best
					yield from deduplicate_external(chain(already_processed, reader), key, sort_key)
					break
	else:
		yield from best.values()


def deduplicate_external(reader: Iterator[TranslationUnit], key: Callable[[TranslationUnit], Any], sort_key: Callable[[TranslationUnit], Any] = lambda unit: 0) -> Iterator[TranslationUnit]:
	best = OrderedDict() # type: Dict[int,List[int]]

	with TemporaryFile() as fh:
		for n, unit in enumerate(reader, start=1):
			offset = fh.tell()

			pickle.dump(unit, fh)

			unit_id = hash(key(unit))

			if unit_id in best:
				best[unit_id].append(offset)
			else:
				best[unit_id] = [offset]

			if n % 10000 == 0:
				mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
				disk_usage = fh.tell()
				info('best contains %d (%d processed) entries (mem: %1.2f GB, disk: %1.2f GB)', len(best), n, mem_usage / 10**9, disk_usage / 10**9)

		info('All entries inspected, %d unique entries; building output', len(best))

		for n, duplicates in enumerate(best.values(), start=1):
			best_unit = TranslationUnit()

			for offset in duplicates:
				fh.seek(offset)
				unit = pickle.load(fh)

				if not best_unit:
					best_unit = unit
				else:
					best_unit = deduplicate_merge(best_unit, unit, sort_key)

			if n % 10000 == 0:
				info('%d out of %d built', n, len(best))

			yield best_unit


def deduplicate_merge(best_unit: TranslationUnit, new_unit: TranslationUnit, sort_key: Callable[[TranslationUnit], Any]) -> TranslationUnit:
	"""Merges new_unit into best_unit, combining collections but overwriting
	all other entries if and only if compare(current, new) is true"""
	new_is_better = sort_key(new_unit) < sort_key(best_unit)

	if new_is_better:
		for key, value in new_unit.items():
			best_unit[key] = value

	for lang, translation in new_unit.translations.items():
		best_unit.translations[lang].updateVariant(translation)

	return best_unit



def translation_unit_test_prop(lhs: str, test: Callable[[Union[str,float,Set[str]]], bool], unit: TranslationUnit) -> bool:
	"""Tests a translation unit property, whether it is inside a translation
	or a unit level property."""
	if lhs in unit:
		return test(unit[lhs])

	for lang, translation_unit in unit.translations.items():
		if lhs == 'text':
			return test(translation_unit.text)
		elif lhs in translation_unit:
			if test(translation_unit[lhs]):
				return True
	else:
		return False


T = TypeVar('T', float, str)

def build_binary_condition(type: Type[T], op: Callable[[T,T], bool]) -> Callable[[str,str], Callable[[TranslationUnit], bool]]:
	"""Wrapper for standard python operations on types. I.e. to implement gt
	and lt."""
	def build_condition(lhs: str, rhs: str) -> Callable[[TranslationUnit], bool]:
		def test(val: Union[float,str,Set[str]]) -> bool:
			if isinstance(val, set):
				return any(op(type(el), type(rhs)) for el in val)
			else:
				return op(type(val), type(rhs))
		return partial(translation_unit_test_prop, lhs, test)
	return build_condition


def build_eq_condition(lhs: str, rhs: str) -> Callable[[TranslationUnit], bool]:
	"""Specialised version of build_binary_condition that uses 'in' for set
	tests instead of iterating over all elements in the set."""
	def test(val: Union[float,str,Set[str]]) -> bool:
		if isinstance(val, set):
			return rhs in val
		else:
			return rhs == str(val)

	return partial(translation_unit_test_prop, lhs, test)


def build_regex_condition(lhs: str, rhs: str) -> Callable[[TranslationUnit], bool]:
	"""Specialised version (or wrapper around) build_binary_condition that makes
	one that tests a regular expression."""
	pattern = re.compile(rhs)

	def test(val: Union[float,str,Set[str]]) -> bool:
		if isinstance(val, set):
			return any(pattern.search(el) is not None for el in val)
		else:
			return pattern.search(str(val)) is not None

	return partial(translation_unit_test_prop, lhs, test)


condition_operators = {
	 '<': build_binary_condition(float, operator.lt),
	 '>': build_binary_condition(float, operator.gt),
	'<=': build_binary_condition(float, operator.le),
	'>=': build_binary_condition(float, operator.ge),
	 '=': build_eq_condition,
	'=~': build_regex_condition
}


def set_property(key: str, value: Any, unit: TranslationUnit) -> TranslationUnit:
	unit[key] = value
	return unit


def del_properties(properties: List[str], unit: TranslationUnit) -> TranslationUnit:
	for prop in properties:
		del unit[prop]
	return unit


def parse_properties(props: str) -> Dict[str,str]:
	return dict(cast(Tuple[str,str], prop.split('=', 1)) for prop in props.split(','))


def parse_condition(operators: Dict[str,Callable[[str,str], Callable[[TranslationUnit], bool]]], expr: str) -> Callable[[TranslationUnit], bool]:
	pattern = r'^(?P<lhs>\w[\-\w]*)(?P<op>{operators})(?P<rhs>.+)$'.format(
		operators='|'.join(re.escape(op) for op in sorted(operators.keys(), key=len, reverse=True)))

	match = re.match(pattern, expr)
	
	if match is None:
		raise ValueError("Could not parse condition '{}'".format(expr))

	info("Using expression op:'%(op)s' lhs:'%(lhs)s' rhs:'%(rhs)s'", match.groupdict())

	return operators[match.group('op')](match.group('lhs'), match.group('rhs'))


def closer(fh: IO[Any]) -> Generator[Any,None,None]:
	"""Generator that closes fh once it it their turn."""
	if fh.close:
		fh.close()
	yield from []


def is_gzipped(fh: BufferedBinaryIO) -> bool:
	"""Test if stream is probably a gzip stream"""
	return fh.peek(2).startswith(b'\x1f\x8b')


def make_reader(fh: BufferedBinaryIO, args: Namespace = Namespace(input_format=None, input_columns=None, input_languages=None)) -> Iterator[TranslationUnit]:
	if is_gzipped(fh):
		fh = cast(BufferedBinaryIO, gzip.open(fh))

	if not args.input_format:
		file_format, format_args = autodetect(fh)
	else:
		file_format, format_args = args.input_format, {}

	if file_format == 'tab' and 'columns' not in format_args and args.input_columns:
		format_args['columns'] = args.input_columns

	text_fh = TextIOWrapper(fh, encoding='utf-8')

	if file_format == 'tmx':
		reader = TMXReader(text_fh) # type: Reader
	elif file_format == 'tab':
		if not args.input_languages or len(args.input_languages) != 2:
			raise ValueError("'tab' format needs exactly two input languages specified")
		
		reader = TabReader(text_fh, *args.input_languages, **format_args)
	else:
		raise ValueError("Cannot create file reader for format '{}'".format(file_format))

	# Hook an empty generator to the end that will close the file we opened.
	return chain(reader, closer(text_fh))


def peek_first_line(fh: BufferedBinaryIO, length: int = 128) -> bytes:
	"""Tries to get the first full line in a buffer that supports peek."""
	while True:
		buf = fh.peek(length)

		pos = buf.find(b'\n')
		if pos != -1:
			return buf[0:pos]

		if len(buf) < length:
			return buf

		buf *= 2


def autodetect(fh: BufferedBinaryIO) -> Tuple[str, Dict[str,Any]]:
	"""Fill in arguments based on what we can infer from the input we're going
	to get. fh needs to have a peek() method and return bytes."""

	# First test: is it XML?
	xml_signature = b'<?xml '
	if fh.peek(len(xml_signature)).startswith(xml_signature):
		return 'tmx', {}
	
	# Second test: Might it be tab-separated content? And if so, how many columns?
	column_count = 1 + peek_first_line(fh).count(b'\t')
	if column_count >= 7:
		return 'tab', {'columns': ['source-document-1', 'source-document-2', 'text-1', 'text-2', 'hash-bifixer', 'score-bifixer', 'score-bicleaner']}

	if column_count >= 5:
		return 'tab', {'columns': ['source-document-1', 'source-document-2', 'text-1', 'text-2', 'score-aligner']}

	raise ValueError('Did not recognize file format')


def make_deduplicator(args: Namespace, reader: Iterator[TranslationUnit], mem_limit=2 * 10**9) -> Iterator[TranslationUnit]:
	"""
	Make a deduplicate filter based on the input options. Fancy bifixer based
	deduplicator if we have the data, otherwise fall back to boring deduplicator.
	"""

	# Grab the first object from the reader to see what we're dealing with
	try:
		peeked_obj = next(reader)
	except StopIteration:
		# It's an empty reader. No need to wrap it in anything deduplicating.
		return reader

	# Stick the peeked object back on :P
	reader = chain([peeked_obj], reader)

	if 'hash-bifixer' in peeked_obj and 'score-bifixer' in peeked_obj:
		return deduplicate(reader, key=operator.itemgetter('hash-bifixer'), sort_key=operator.itemgetter('score-bifixer'), mem_limit=mem_limit)
	else:
		return deduplicate(reader, key=text_key, mem_limit=mem_limit)


def abort(message: str) -> int:
	"""Abandon ship! Use in case of misguided users."""
	print(message, file=sys.stderr)
	return 1


def properties_adder(properties: Dict[str,str], reader: Iterator[TranslationUnit]) -> Iterator[TranslationUnit]:
	for unit in reader:
		unit.update(properties)
		yield unit


def main(argv: List[str], stdin: TextIO, stdout: TextIO) -> int:
	parser = ArgumentParser(description='Annotate, filter and convert tmx files')
	parser.add_argument('-i', '--input-format', choices=['tmx', 'tab'], help='Input file format. Automatically detected if left unspecified.')
	parser.add_argument('-o', '--output-format', choices=['tmx', 'tab', 'txt', 'py'], default='tmx', help='Output file format. Output is always written to stdout.')
	parser.add_argument('-l', '--input-languages', nargs=2, help='Input languages in case of tab input. Needs to be in order their appearance in the columns.')
	parser.add_argument('-c', '--input-columns', nargs='+', help='Input columns in case of tab input. Column names ending in -1 or -2 will be treated as translation-specific.')
	parser.add_argument('--output-languages', nargs='+', help='Output languages for tab and txt output. txt output allows only one language, tab multiple.')
	parser.add_argument('--creation-date', type=fromisoformat, default=datetime.now(), help='override creation date in tmx output.')
	parser.add_argument('-p', '--properties', action='append', help='List of A=B,C=D properties to add to each sentence pair. You can use one --properties for all files or one for each input file.')
	parser.add_argument('-d', '--deduplicate', action='store_true', help='Deduplicate units before printing. Unit properties are combined where possible. If score-bifixer and hash-bifixer are avaiable, these will be used.')
	parser.add_argument('--drop', nargs='+', dest='drop_properties', help='Drop properties from output.')
	parser.add_argument('--renumber-output', action='store_true', help='Renumber the translation unit ids. Always enabled when multiple input files are given.')
	parser.add_argument('--ipc', dest='ipc_meta_files', action='append', type=FileType('r'), help='One or more IPC metadata files.')
	parser.add_argument('--ipc-group', dest='ipc_group_files', action='append', type=FileType('r'), help='One or more IPC grouping files.')
	parser.add_argument('--with', nargs='+', action='append', dest='filter_with')
	parser.add_argument('--without', nargs='+', action='append', dest='filter_without')
	parser.add_argument('--verbose', action='store_true', help='Print progress updates.')
	parser.add_argument('--workspace', type=fromfilesize, help='Mamimum memory usage for deduplication. When exceeded, will continue deduplication using filesystem.', default='4G')
	parser.add_argument('files', nargs='*', default=[stdin.buffer], type=FileType('rb'), help='Input files. May be gzipped. If not specified stdin is used.')

	# I prefer the modern behaviour where you can do `tmxutil.py -p a=1 file.tmx
	# -p a=2 file2.tmx` etc. but that's only available since Python 3.7.
	if hasattr(parser, 'parse_intermixed_args'):
		args = parser.parse_intermixed_args(argv)
	else:
		args = parser.parse_args(argv)

	fout = stdout

	if args.verbose:
		getLogger().setLevel(INFO)

	# Create reader. Make sure to call make_reader immediately and not somewhere
	# down in a nested generator so if one of the files cannot be found, we
	# error out immediately.
	readers = [make_reader(fh, args) for fh in args.files]

	# Add properties to each specific file? If so, do it before we chain all
	# readers into a single iterator. If all share the same properties we'll
	# add it after chaining multiple readers into one.
	if args.properties and len(args.properties) > 1:
		if len(args.properties) != len(readers):
			return abort("When specifying multiple --properties options, you need"
			             " to specify exactly one for each input file. You have {}"
			             " --properties options, but {} files.".format(len(args.properties), len(readers)))
		properties_per_file = (parse_properties(props) for props in args.properties)

		readers = [properties_adder(properties, reader) for properties, reader in zip(properties_per_file, readers)]
		
		# If we have multiple input files, the translation unit ids will be a mess
		# when merged. So renumber them.
		args.renumber_output = True

	# Merge all readers into a single source of sentence pairs
	reader = chain.from_iterable(readers)

	# If we want to add properties (the same ones) to all input files, we do it
	# now, after merging all readers into one.
	if args.properties and len(args.properties) == 1:
		properties = parse_properties(args.properties[0])
		reader = properties_adder(properties, reader)

	# Create writer
	if args.output_format == 'tmx':
		writer = TMXWriter(fout, creation_date=args.creation_date) # type: Writer
	elif args.output_format == 'tab':
		writer = TabWriter(fout, args.output_languages)
	elif args.output_format == 'txt':
		if not args.output_languages or len(args.output_languages) != 1:
			return abort("Use --output-languages X to select which language."
			             " When writing txt, it can only write one language at"
			             " a time.")
		writer = TxtWriter(fout, args.output_languages[0])
	elif args.output_format == 'py':
		writer = PyWriter(fout)
	else:
		raise ValueError('Unknown output format: {}'.format(args.output_format))

	# Optional filter & annotation steps for reader.

	if args.ipc_meta_files:
		reader = map(IPCLabeler(args.ipc_meta_files).annotate, reader)

	if args.ipc_group_files:
		reader = map(IPCGroupLabeler(args.ipc_group_files).annotate, reader)

	if args.filter_with:
		dnf = [[parse_condition(condition_operators, cond_str) for cond_str in cond_expr] for cond_expr in args.filter_with]
		reader = filter(lambda unit: any(all(expr(unit) for expr in cond) for cond in dnf), reader)

	if args.filter_without:
		dnf = [[parse_condition(condition_operators, cond_str) for cond_str in cond_expr] for cond_expr in args.filter_without]
		reader = filter(lambda unit: all(any(not expr(unit) for expr in cond) for cond in dnf), reader)

	if args.deduplicate:
		reader = make_deduplicator(args, reader, mem_limit=args.workspace)

	if args.renumber_output:
		reader = starmap(partial(set_property, 'id'), enumerate(reader, start=1))

	# If we want to drop properties from the output, do that as the last step.
	if args.drop_properties:
		reader = map(partial(del_properties, args.drop_properties), reader)

	# Main loop. with statement for writer so it can write header & footer
	with writer:
		count = 0
		for unit in reader:
			writer.write(unit)
			count += 1
		info("Written %d records.", count)

	return 0


if __name__ == '__main__':
	try:
		sys.exit(main(sys.argv[1:], sys.stdin, sys.stdout))
	except ValueError as e:
		sys.exit(abort("Error: {}".format(e)))
