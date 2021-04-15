#!/usr/bin/env python3
# Tool to convert between tab, txt and tmx formatting, filtering & adding
# annotations.

__VERSION__ = 1.1

import csv
import sys
import os
import re
import gzip
import pickle
import resource
import operator
import importlib.util
from abc import ABC, ABCMeta, abstractmethod
from argparse import ArgumentParser, FileType, Namespace
from collections import defaultdict, OrderedDict, Counter
from contextlib import contextmanager
from datetime import datetime
from functools import partial
from io import BufferedReader, TextIOWrapper
from itertools import combinations, chain, starmap
from functools import reduce
from logging import info, warning, getLogger, INFO
from math import floor
from operator import itemgetter
from pprint import pprint
from tempfile import TemporaryFile
from time import time
from typing import Callable, Dict, List, Counter, Optional, Any, Iterator, Iterable, Set, FrozenSet, Tuple, Type, TypeVar, BinaryIO, TextIO, IO, Union, cast, Generator, Sequence, Mapping
from types import TracebackType
from xml.sax.saxutils import escape, quoteattr
from xml.etree.ElementTree import iterparse, Element

try:
	from tqdm.autonotebook import tqdm
except ImportError:
	tqdm = None

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


class TranslationUnit(Dict[str,Set[str]]):
	__slots__ = ['translations']

	def __init__(self, *, translations: Optional[Dict[str, TranslationUnitVariant]] = None, **kwargs: Set[str]):
		super().__init__(**kwargs)
		self.translations = translations or dict() # type: Dict[str,TranslationUnitVariant]


class BufferedBinaryIO(BinaryIO, metaclass=ABCMeta):
	@abstractmethod
	def peek(self, size: int) -> bytes:
		...


class ProgressWrapper:
	"""Wraps around a file-like object and shows a progress bar as to how much
	of it has been read."""

	def __init__(self, fh: Any):
		self.fh = fh
		self.tqdm = tqdm(
			desc=fh.name,
			total=os.fstat(fh.fileno()).st_size,
			initial=fh.seekable() and fh.tell(),
			file=sys.stderr,
			unit='b',
			unit_scale=True)

	def __getattr__(self, attr: str) -> Any:
		return getattr(self.fh, attr)

	def read(self, size: int = -1) -> Any:
		data = self.fh.read(size)
		self.tqdm.update(len(data))
		return data

	def read1(self, size: int = -1) -> Any:
		data = self.fh.read1(size)
		self.tqdm.update(len(data))
		return data

	def close(self) -> None:
		self.tqdm.close()
		self.fh.close()


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

	def __init__(self, fh: BinaryIO):
		self.fh = fh

	def close(self) -> None:
		self.fh.close()

	def records(self) -> Iterator[TranslationUnit]:
		stack = list() # type: List[Element]
		path = list() # type: List[str]
		
		info("TMXReader starts reading from %s", self.fh.name)
		
		unit: TranslationUnit
		translation: TranslationUnitVariant

		lang_key = '{http://www.w3.org/XML/1998/namespace}lang'
		
		for event, element in iterparse(self.fh, events=('start', 'end')):
			if event == 'start':
				stack.append(element)
				path.append(element.tag)

				if path == ['tmx', 'body', 'tu']:
					unit = TranslationUnit(id={element.get('tuid')})
				elif path == ['tmx', 'body', 'tu', 'tuv']:
					translation = TranslationUnitVariant()
			elif event == 'end':
				if path == ['tmx', 'body', 'tu']:
					yield unit
				elif path == ['tmx', 'body', 'tu', 'prop']:
					if element.text is None:
						warning('empty <prop type="%s"></prop> encountered in unit with id %s in file %s; property ignored', element.get('type'), first(unit['id']), self.fh.name)
					else:
						unit.setdefault(element.get('type'), set()).add(element.text.strip())
				elif path == ['tmx', 'body', 'tu', 'tuv']:
					unit.translations[element.attrib[lang_key]] = translation
					translations = None
				elif path == ['tmx', 'body', 'tu', 'tuv', 'prop']:
					if element.text is None:
						warning('empty <prop type="%s"></prop> encountered in unit with id %s in file %s; property ignored', element.get('type'), first(unit['id']), self.fh.name)
					else:
						translation.setdefault(element.get('type'), set()).add(element.text.strip())
				elif path == ['tmx', 'body', 'tu', 'tuv', 'seg']:
					if element.text is None:
						warning('empty translation segment encountered in unit with id %s in file %s', first(unit['id']), self.fh.name)
						translation.text = ''
					else:
						translation.text = element.text.strip()

				path.pop()
				stack.pop()
				if stack:
					stack[-1].remove(element)


# _escape_cdata and _escape_attrib are copied from 
# https://github.com/python/cpython/blob/3.9/Lib/xml/etree/ElementTree.py
def _escape_cdata(text: str) -> str:
	if "&" in text:
		text = text.replace("&", "&amp;")
	if "<" in text:
		text = text.replace("<", "&lt;")
	if ">" in text:
		text = text.replace(">", "&gt;")
	return text


def _escape_attrib(text: str) -> str:
	if "&" in text:
		text = text.replace("&", "&amp;")
	if "<" in text:
		text = text.replace("<", "&lt;")
	if ">" in text:
		text = text.replace(">", "&gt;")
	if "\"" in text:
		text = text.replace("\"", "&quot;")
	if "\r" in text:
		text = text.replace("\r", "&#13;")
	if "\n" in text:
		text = text.replace("\n", "&#10;")
	if "\t" in text:
		text = text.replace("\t", "&#09;")
	return text


def _flatten(unit: Mapping[str,Set[str]]) -> Iterator[Tuple[str,str]]:
	for key, values in unit.items():
		for value in values:
			yield key, value


class TMXWriter(Writer):
	def __init__(self, fh: TextIO, *, creation_date: Optional[datetime] = None):
		self.fh = fh
		self.creation_date = creation_date
		
	def __enter__(self) -> 'TMXWriter':
		self.fh.write('<?xml version="1.0"?>\n'
					  '<tmx version="1.4">\n'
					  '  <header\n'
					  '    o-tmf="PlainText"\n'
					  '    creationtool="tmxutil"\n'
					  '    creationtoolversion="' + str(__VERSION__) + '"\n'
					  '    datatype="PlainText"\n'
					  '    segtype="sentence"\n'
					  '    o-encoding="utf-8"\n'
					  '    creationdate="' + self.creation_date.strftime("%Y%m%dT%H%M%S") + '">\n'
					  '  </header>\n'
					  '  <body>\n')
		return self

	def __exit__(self, type: Optional[Type[BaseException]], value: Optional[BaseException], traceback: Optional[TracebackType]) -> None:
		self.fh.write('  </body>\n'
					        '</tmx>\n')

	def write(self, unit: TranslationUnit) -> None:
		self.fh.write('    <tu tuid="' + _escape_attrib(str(first(unit['id']))) + '" datatype="Text">\n')
		for key, value in sorted(_flatten(unit)):
			if key != 'id':
				self.fh.write('      <prop type="' + _escape_attrib(str(key)) + '">' + _escape_cdata(str(value)) + '</prop>\n')
		
		for lang, translation in sorted(unit.translations.items()):
			self.fh.write('      <tuv xml:lang="' + _escape_attrib(lang) + '">\n')
			for key, value in sorted(_flatten(translation)):
				self.fh.write('        <prop type="' + _escape_attrib(str(key)) + '">' + _escape_cdata(str(value)) + '</prop>\n')
			self.fh.write('        <seg>' + _escape_cdata(str(translation.text)) + '</seg>\n'
			              '      </tuv>\n')
		self.fh.write('    </tu>\n')
			

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

			values = line.split('\t')

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


class CountWriter(Writer):
	"""Instead of writing tmx records, it counts a property and writes a summary
	of which values it encountered for that property, and how often it encountered
	them."""
	def __init__(self, fh: TextIO, key: Callable[[TranslationUnit], List[Any]]):
		self.fh = fh
		self.key = key

	def __enter__(self) -> 'CountWriter':
		self.counter = Counter() # type: Counter[Any]
		return self

	def __exit__(self, type: Optional[Type[BaseException]], value: Optional[BaseException], traceback: Optional[TracebackType]) -> None:
		if type is None:
			for key, count in sorted(self.counter.most_common(), key=itemgetter(1), reverse=True):
				self.fh.write("{}\t{}\n".format(count, " ".join(sorted(key)) if isinstance(key, frozenset) else key))

	def write(self, unit: TranslationUnit) -> None:
		self.counter.update(self.key(unit))


class LiveCountWriter(Writer):
	"""Live variant of CountWriter: shows live updating bars while counting."""
	def __init__(self, fh: TextIO, key: Callable[[TranslationUnit], List[Any]]):
		self.fh = fh
		self.key = key
		self.top_n = 10

	def __enter__(self) -> 'CountWriter':
		self.counter = Counter()
		self.total = 0
		self.bars = []
		self.n = 0
		self.last_update = time()
		self.last_n = 0
		self.update_interval = 128
		return self

	def __exit__(self, type: Optional[Type[BaseException]], value: Optional[BaseException], traceback: Optional[TracebackType]) -> None:
		for bar in self.bars:
			bar.close()

		if type is None:
			for key, count in self.counter.most_common():
				self.fh.write("{}\t{}\n".format(count, " ".join(sorted(key)) if isinstance(key, frozenset) else key))

	def refresh(self):
		top = self.counter.most_common(self.top_n)
		remainder = len(self.counter) - len(top)

		if remainder:
			remainder_count = self.total - sum(count for _, count in top)
			top.append(('({} more)'.format(remainder), remainder_count))

		# Make sure we've enough bars
		while len(top) > len(self.bars):
			self.bars.append(tqdm(
				position=len(self.bars)+1,
				unit='unit',
				file=sys.stderr,
				dynamic_ncols=True,
				bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'))

		# Determine the label length for alignment
		label_len=max(len(str(value)) for value, _ in top)

		# Update all bars (sorting by most common on top)
		for bar, (value, count) in zip(self.bars, top):
			bar.set_description('{{: <{:d}s}}'.format(label_len+2).format(str(value)), refresh=False)
			bar.total = self.total
			bar.n = count
			bar.refresh()

	def _count_iter(self, iterable):
		count = 0
		for item in iterable:
			count += 1
			yield item
		self.total += count

	def _smooth(self, current, target):
		return floor(0.7 * current + 0.3 * target)

	def write(self, unit: TranslationUnit) -> None:
		vals = self.key(unit)
		self.counter.update(self._count_iter(vals))
		self.n += 1
		if self.n % self.update_interval == 0:
			time_since_last_update = max(time() - self.last_update, 1e-10)
			n_per_sec = self.update_interval / time_since_last_update
			self.update_interval = max(self._smooth(self.update_interval, 0.5 * n_per_sec), 1)
			self.last_update = time()
			self.refresh()


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

			best.setdefault(unit_id, []).append(offset)

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



def translation_unit_test_prop(lhs: str, test: Callable[[Set[str]], bool], unit: TranslationUnit) -> bool:
	"""Tests a translation unit property, whether it is inside a translation
	or a unit level property."""
	if lhs in unit:
		return test(unit[lhs])

	for lang, translation_unit in unit.translations.items():
		if lhs == 'text':
			return test({translation_unit.text})
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
		def test(val: Set[str]) -> bool:
			return any(op(type(el), type(rhs)) for el in val)
		return partial(translation_unit_test_prop, lhs, test)
	return build_condition


def build_eq_condition(lhs: str, rhs: str) -> Callable[[TranslationUnit], bool]:
	"""Specialised version of build_binary_condition that uses 'in' for set
	tests instead of iterating over all elements in the set."""
	def test(val: Set[str]) -> bool:
		return rhs in val
	return partial(translation_unit_test_prop, lhs, test)


def build_regex_condition(lhs: str, rhs: str) -> Callable[[TranslationUnit], bool]:
	"""Specialised version (or wrapper around) build_binary_condition that makes
	one that tests a regular expression."""
	pattern = re.compile(rhs)
	def test(val: Set[str]) -> bool:
		return any(pattern.search(el) is not None for el in val)
	return partial(translation_unit_test_prop, lhs, test)


condition_operators = {
	 '<': build_binary_condition(float, operator.lt),
	 '>': build_binary_condition(float, operator.gt),
	'<=': build_binary_condition(float, operator.le),
	'>=': build_binary_condition(float, operator.ge),
	 '=': build_eq_condition,
	'=~': build_regex_condition
}


def set_property(key: str, value: str, unit: TranslationUnit) -> TranslationUnit:
	unit[key] = {value}
	return unit


def del_properties(properties: List[str], unit: TranslationUnit) -> TranslationUnit:
	for prop in properties:
		del unit[prop]
	return unit


def parse_properties(props: str) -> Dict[str,Set[str]]:
	properties: Dict[str,Set[str]] = {}
	for prop in props.split(','):
		key, value = prop.split('=', 1)
		properties.setdefault(key, set()).add(value)
	return properties


def parse_condition(operators: Mapping[str,Callable[[str,str], Callable[[TranslationUnit], bool]]], expr: str) -> Callable[[TranslationUnit], bool]:
	pattern = r'^(?P<lhs>\w[\-\w]*)(?P<op>{operators})(?P<rhs>.+)$'.format(
		operators='|'.join(re.escape(op) for op in sorted(operators.keys(), key=len, reverse=True)))

	match = re.match(pattern, expr)
	
	if match is None:
		raise ValueError("Could not parse condition '{}'".format(expr))

	info("Using expression op:'%(op)s' lhs:'%(lhs)s' rhs:'%(rhs)s'", match.groupdict())

	return operators[match.group('op')](match.group('lhs'), match.group('rhs'))


def closer(fh: Any) -> Generator[Any,None,None]:
	"""Generator that closes fh once it it their turn."""
	if hasattr(fh, 'close'):
		fh.close()
	yield from []


def is_gzipped(fh: BufferedBinaryIO) -> bool:
	"""Test if stream is probably a gzip stream"""
	return fh.peek(2).startswith(b'\x1f\x8b')


def make_reader(fh: BufferedBinaryIO, *, input_format: Optional[str] = None, input_columns: Optional[Iterable[str]] = None, input_languages: Optional[Sequence[str]] = None, progress:bool = False, **kwargs: Any) -> Iterator[TranslationUnit]:
	if tqdm and progress:
		fh = ProgressWrapper(fh)

	if is_gzipped(fh):
		fh = cast(BufferedBinaryIO, gzip.open(fh))

	if not input_format:
		file_format, format_args = autodetect(fh)
	else:
		file_format, format_args = input_format, {}

	if file_format == 'tab' and 'columns' not in format_args and input_columns:
		format_args['columns'] = input_columns

	if file_format == 'pickle':
		reader: Reader = PickleReader(fh)
	elif file_format == 'tmx':
		reader = TMXReader(fh)
	elif file_format == 'tab':
		if not input_languages or len(input_languages) != 2:
			raise ValueError("'tab' format needs exactly two input languages specified")
		text_fh = TextIOWrapper(fh, encoding='utf-8')
		reader = TabReader(text_fh, *input_languages, **format_args)
	else:
		raise ValueError("Cannot create file reader for format '{}'".format(file_format))

	# Hook an empty generator to the end that will close the file we opened.
	return chain(reader, closer(reader))


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


def first_item_getter(key: str) -> Callable[[TranslationUnit], Optional[str]]:
	"""Creates a getter that gets one value from a translation unit's properties,
	if there are more values for that property, it's undefined which one it gets.
	If the property does not exist, or is empty, it will return None."""
	def getter(obj: TranslationUnit) -> Optional[str]:
		return first(obj.get(key, set()), default=None)
	return getter


def make_deduplicator(args: Namespace, reader: Iterator[TranslationUnit], mem_limit : int = 2 * 10**9) -> Iterator[TranslationUnit]:
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
		return deduplicate(reader, key=first_item_getter('hash-bifixer'), sort_key=first_item_getter('score-bifixer'), mem_limit=mem_limit)
	else:
		return deduplicate(reader, key=text_key, mem_limit=mem_limit)


def abort(message: str) -> int:
	"""Abandon ship! Use in case of misguided users."""
	print(message, file=sys.stderr)
	return 1


def properties_adder(properties: Dict[str,Set[str]], reader: Iterator[TranslationUnit]) -> Iterator[TranslationUnit]:
	for unit in reader:
		unit.update(properties)
		yield unit


def import_file_as_module(file):
	filename = os.path.basename(file)
	basename, ext = os.path.splitext(filename)
	if ext not in {'.py'}:
		raise ValueError('Error importing {}: can only import .py files'.format(file))

	spec = importlib.util.spec_from_file_location(basename, file)
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


def concat_object(a, b):
	out = object()

	for module in a, b:
		for attr in dir(module):
			setattr(out, attr, getattr(module, attr))

	return out


def parse_count_property(expr: str, library: Dict[str,Callable[Any,Any]] = {'len': len}) -> Callable[[TranslationUnit], Iterable[Any]]:
	ops = []

	while True:
		match = re.match(r'^(?P<fun>[a-zA-Z_]\w*)\((?P<expr>.+?)\)$', expr)
		if not match:
			break
		ops.append(library[match.group('fun')])
		expr = match.group('expr')

	match = re.match(r'^((?P<lang>[\w-]+)?(?P<dot>\.))?(?P<prop>[\w-]+)(?P<brackets>\[\])?$', expr)
	if not match:
		raise ValueError('Could not interpret count expression `{}`'.format(expr))

	prop = match.group('prop')

	# 'en.source-document' or 'en.text'
	if match.group('lang'):
		lang = match.group('lang')	
		if prop == 'text':
			val_getter = lambda unit: [unit.translations[lang].text]
		else:
			val_getter = lambda unit: unit.translations[lang][prop]
	# e.g. '.collection', only look in root
	elif match.group('dot'):
		val_getter = lambda unit: unit[prop]
	# e.g. 'text'; text can only occur in translations
	elif prop == 'text':
		val_getter = lambda unit: (translation.text for translation in unit.translations.values())
	# e.g. 'source-document' or 'collection'; search through both root and translations
	else:
		val_getter = lambda unit: reduce(lambda acc, translation: acc + list(translation.get(prop, [])), unit.translations.values(), list(unit.get(prop, [])))

	if match.group('brackets'):
		agg_getter = lambda unit: [frozenset(val_getter(unit))] # convert to frozenset so it can be used as key in dict/Counter
	else:
		agg_getter = val_getter

	if ops:
		fun_getter = lambda unit: (reduce(lambda val, op: op(val), ops, val) for val in agg_getter(unit))
	else:
		fun_getter = agg_getter

	return fun_getter


def main(argv: List[str], stdin: TextIO, stdout: TextIO) -> int:
	parser = ArgumentParser(description='Annotate, filter and convert tmx files')
	parser.add_argument('-i', '--input-format', choices=['tmx', 'tab', 'pickle'], help='Input file format. Automatically detected if left unspecified.')
	parser.add_argument('-o', '--output-format', choices=['tmx', 'tab', 'txt', 'py', 'pickle'], default='tmx', help='Output file format. Output is always written to stdout.')
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
	parser.add_argument('-P', '--progress', action='store_true', help='Show progress bar when reading files')
	parser.add_argument('--verbose', action='store_true', help='Print progress updates.')
	parser.add_argument('--workspace', type=fromfilesize, help='Mamimum memory usage for deduplication. When exceeded, will continue deduplication using filesystem.', default='4G')
	parser.add_argument('--count', dest='count_property', help='Count which values occur for a property. E.g. `.collection` (count every collection observed), `source-document`, `len(en.text)`, `.collection[]`.')
	parser.add_argument('--include', action='append', default=[], dest='count_libraries', help='Include a python file so functions defined in that file can be used with --count, e.g. include something that provides a domain(url:str) function, and use `--count domain(source-document)`.')
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
	readers = [make_reader(fh, **vars(args)) for fh in args.files]

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

	# Create writer
	if args.count_property:
		count_property = parse_count_property(args.count_property,
			reduce(lambda obj, file: {**obj, **import_file_as_module(file).__dict__},
				args.count_libraries,
				{'len': len}))

		if tqdm and args.progress:
			writer = LiveCountWriter(fout, key=count_property)
		else:
			writer = CountWriter(fout, key=count_property)
	elif args.output_format == 'tmx':
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
	elif args.output_format == 'pickle':
		writer = PickleWriter(fout.buffer)
	else:
		raise ValueError('Unknown output format: {}'.format(args.output_format))

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
