#!/usr/bin/env python3
# Tool to convert between tab, txt and tmx formatting, filtering & adding
# annotations.

__VERSION__ = 1.1

import csv
import sys
import time
import re
import gzip
from abc import ABC, abstractmethod
from argparse import ArgumentParser, FileType, Namespace
from collections import defaultdict
from contextlib import contextmanager
from io import BufferedReader, TextIOWrapper
from itertools import combinations, chain
from operator import itemgetter
from pprint import pprint
from typing import Callable, Dict, List, Optional, Any, Iterator, Set, Tuple, BinaryIO, TextIO, IO, cast
from xml.sax.saxutils import escape, quoteattr
from xml.etree.ElementTree import iterparse


class BufferedBinaryIO(BinaryIO):
	def peek(self, size: int) -> bytes:
		pass


class XMLWriter(object):
	"""Writes XML. Light wrapper around iobase.write really, but handles
	properly indenting and closing xml elements at the right time."""

	def __init__(self, fh):
		self.fh = fh
		self.stack = []
		self.indent = '  '

	def open(self, name: str, attributes: dict = dict()):
		if self.stack:
			self.stack[-1] = (self.stack[-1][0], True)

		attr_str = ''.join(' {}={}'.format(attr_name, quoteattr(str(attr_value))) for attr_name, attr_value in attributes.items())
		self.fh.write('\n{}<{}{}>'.format(self.indent * len(self.stack), name, attr_str))

		self.stack.append((name, False))

	def close(self):
		name, has_children = self.stack.pop()
		if has_children:
			self.fh.write('\n{}</{}>'.format(self.indent * len(self.stack), name))
		else:
			self.fh.write('</{}>'.format(name))

	def write(self, text: Any):
		self.fh.write(escape(str(text).rstrip()))

	@contextmanager
	def element(self, name: str, attributes: dict = dict()):
		self.open(name, attributes)
		yield self
		self.close()

	def __enter__(self) -> 'XMLWriter':
		self.fh.write('<?xml version=\"1.0\"?>')
		return self

	def __exit__(self, type, value, traceback):
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

	def __iter__(self) -> Iterator[dict]:
		return self.records()

	@abstractmethod
	def records(self) -> Iterator[dict]:
		pass


class Writer(ABC):
	"""Interface for sentence pair output stream. Has with statement context
	magic functions that can be overwritten to deal with writing headers and
	footers, or starting and ending XML output."""

	def __enter__(self):
		return self

	def __exit__(self, *args):
		pass

	@abstractmethod
	def write(self, unit: dict):
		pass


class TMXReader(Reader):
	"""TMX File format reader. XML attributes are mostly ignored. <prop/>
	elements of the <tu/> are added as attributes, and of <tuv/> as attributes
	with sets of values as we expect one or more of them, i.e. one or more
	source-document, ipc, etc."""

	def __init__(self, fh):
		self.fh = fh

	def records(self) -> Iterator[dict]:
		path = []
		stack = []
		
		def enter(element):
			stack.append(element)
			path.append(element.tag)

		def exit(element):
			removed = stack.pop()
			assert removed == element
			path.pop()
			if stack:
				# Remove element from parent to keep the internal tree empty
				stack[-1].remove(removed)

		unit: dict = {}
		translation: dict = {}

		lang_key = '{http://www.w3.org/XML/1998/namespace}lang'
		
		for event, element in iterparse(self.fh, events=('start', 'end')):
			if event == 'start':
				enter(element)

				if path == ['tmx', 'body', 'tu']:
					unit = {
						'id': element.get('tuid'),
						'translations': {}
					}
				elif path == ['tmx', 'body', 'tu', 'tuv']:
					translation = defaultdict(set)
			elif event == 'end':
				if path == ['tmx', 'body', 'tu']:
					yield unit
				elif path == ['tmx', 'body', 'tu', 'prop']:
					unit[element.get('type')] = float(element.text.strip()) if 'score' in element.get('type') else element.text.strip()
				elif path == ['tmx', 'body', 'tu', 'tuv']:
					unit['translations'][element.attrib[lang_key]] = translation
					translations = None
				elif path == ['tmx', 'body', 'tu', 'tuv', 'prop']:
					translation[element.get('type')].add(element.text.strip())
				elif path == ['tmx', 'body', 'tu', 'tuv', 'seg']:
					translation['text'] = element.text.strip()

				exit(element)


class TMXWriter(Writer):
	def __init__(self, fh):
		self.fh = fh
		
	def __enter__(self):
		self.writer = XMLWriter(self.fh)
		self.writer.__enter__()
		self.writer.open('tmx', {'version': 1.4})
		with self.writer.element('header', {
			'o-tmf': 'PlainText',
			'creationtool': 'tab2tmx.py',
			'creationtoolversion': __VERSION__,
			'datatype': 'PlainText',
			'segtype': 'sentence',
			'creationdate': time.strftime("%Y%m%dT%H%M%S"),
			'o-encoding': 'utf-8'
			}) as header:
			pass
		self.writer.open('body')
		return self

	def __exit__(self, *args):
		self.writer.__exit__(*args)

	def _write_prop(self, name, value):
		if value is None:
			return
		elif isinstance(value, (list, set)):
			for val in value:
				self._write_prop(name, val)
		else:
			with self.writer.element('prop', {'type': name}) as prop:
				prop.write(value)

	def write(self, unit):
		with self.writer.element('tu', {'tuid': unit['id'], 'datatype': 'Text'}):
			for key, value in unit.items():
				if key not in {'id', 'translations'}:
					self._write_prop(key, value)
			for lang, translation in unit['translations'].items():
				with self.writer.element('tuv', {'xml:lang': lang}):
					for key, value in translation.items():
						if key not in {'text', 'lang'}:
							self._write_prop(key, value)
					with self.writer.element('seg'):
						self.writer.write(translation['text'])


class TabReader(Reader):
	def __init__(self, fh, src_lang, trg_lang, columns=['source-document-1', 'source-document-2', 'text-1', 'text-2', 'score-aligner']):
		self.fh = fh
		self.src_lang = src_lang
		self.trg_lang = trg_lang
		self.columns = columns

	def records(self) -> Iterator[dict]:
		for n, line in enumerate(self.fh):
			values = line.split('\t')

			record = {
				'id': n
			}

			translation1 = {
				'lang': self.src_lang
			}

			translation2 = {
				'lang': self.trg_lang
			}

			for column, value in zip(self.columns, values):
				if column == '-':
					continue

				if column.endswith('-1') or column.endswith('-2'):
					unit = translation1 if column.endswith('-1') else translation2
					unit[column[:-2]] = value if column[:-2] in {'text', 'lang'} else {value}
				else:
					record[column] = value

			yield {
				**record,
				'translations': {
					translation1['lang']: translation1,
					translation2['lang']: translation2
				}
			}


class TabWriter(Writer):
	def __init__(self, fh, languages=[]):
		self.fh = fh
		self.languages = languages

	def __enter__(self):
		self.writer = csv.writer(self.fh, delimiter='\t')

	def write(self, unit: dict):
		if not self.languages:
			self.languages = list(unit['translations'].keys())

		self.writer.writerow(
			  [next(iter(unit['translations'][lang]['source-document'])) for lang in self.languages]
			+ [unit['translations'][lang]['text'] for lang in self.languages])


class TxtWriter(Writer):
	def __init__(self, fh, language: str):
		self.fh = fh
		self.language = language

	def write(self, unit: dict):
		print(unit['translations'][self.language]['text'], file=self.fh)


class PyWriter(Writer):
	def __init__(self, fh):
		self.fh = fh

	def write(self, unit:dict):
		pprint(unit, stream=self.fh)


class IPCLabeler(object):
	"""Add IPC labels to sentence pairs based on the patent ids found in the
	source-document property of either side of the pair."""

	lut: Dict[Tuple[str,str], Set[str]]

	def __init__(self, files: List[TextIO] = []):
		self.lut = dict()
		for fh in files:
			self.load(fh)

	def load(self, fh: TextIO):
		for line in fh:
			src_id, _, _, _, src_lang, src_ipcs, trg_id, _, _, _, trg_lang, trg_ipcs = line.split('\t', 11)
			self.lut[(src_lang.lower(), src_id)] = set(ipc.strip() for ipc in src_ipcs.split(','))
			self.lut[(trg_lang.lower(), trg_id)] = set(ipc.strip() for ipc in trg_ipcs.split(','))

	def annotate(self, unit: dict) -> dict:
		for lang, translation in unit['translations'].items():
			keys = self.lut.keys() & {(lang.lower(), url) for url in translation['source-document']}
			# Ignoring type because https://github.com/python/mypy/issues/2013
			translation['ipc'] = set().union(*(self.lut[key] for key in keys)) # type: ignore
		return unit


class IPCGroupLabeler(object):
	"""Add overall IPC group ids based on IPC labels added by IPCLabeler."""

	patterns: List[Tuple[str,Set[str]]]

	def __init__(self, files: List[TextIO] = []):
		self.patterns = []
		for fh in files:
			self.load(fh)

	def load(self, fh: TextIO):
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

	def annotate(self, unit: dict) -> dict:
		for lang, translation in unit['translations'].items():
			translation['ipc-group'] = set().union(*map(self.find_group, translation['ipc'])) # type: ignore
		return unit


def text_key(unit: dict) -> tuple:
	return tuple(translation['text'] for translation in unit['translations'].values())


def deduplicate(reader: Iterator[dict], key: Callable[[dict], Any], compare: Callable[[dict, dict], bool] = lambda curr, new: False) -> Iterator[dict]:
	"""
	Deduplicate records read from reader. It does this by creating a hash table
	of all records, grouped by key(record). If multiple records have the same
	key they are combined if properties allow this (i.e. sets, lists) or
	overwritten in case compare(current, new) is True. See deduplicate_merge().
	
	Note: This function behaves like an iterator but will only start yielding
	results once reader has run out of records.
	"""

	best: dict = dict()

	for unit in reader:
		unit_id = hash(key(unit))

		if unit_id in best:
			best[unit_id] = deduplicate_merge(best[unit_id], unit, compare)
		else:
			best[unit_id] = unit
	
	yield from best.values()


def deduplicate_merge(best_unit: dict, new_unit: dict, compare: Callable[[dict, dict], bool]) -> dict:
	"""Merges new_unit into best_unit, combining collections but overwriting
	all other entries if and only if compare(current, new) is true"""
	new_is_better = compare(best_unit, new_unit)

	if new_is_better:
		for key, value in new_unit.items():
			if key != 'translations':
				best_unit[key] = value

	for lang, translation in new_unit['translations'].items():
		for t_key, t_value in translation.items():
			if isinstance(t_value, set):
				best_unit['translations'][lang][t_key] |= t_value
			elif isinstance(t_value, list):
				best_unit['translations'][lang][t_key] += t_value
			elif new_is_better: # Text etc, where we need to pick the best instead
			                    # of keep everything but only if it is better
				best_unit['translations'][lang][t_key] = t_value

	return best_unit


def pred_prop_intersection(key: str, values: Set[str]) -> Callable[[dict], bool]:
	return lambda unit: bool(unit[key] & values)


def pred_translation_prop_intersection(key: str, values: Set[str]) -> Callable[[dict], bool]:
	return lambda unit: any(translation[key] & values for translation in unit['translations'].values())


def pred_translation_text_contains(values: List[str]) -> Callable[[dict], bool]:
	pattern = re.compile('|'.join('({})'.format(re.escape(value)) for value in values))
	return lambda unit: any(pattern.search(translation['text']) is not None for translation in unit['translations'].values())


def pred_negate(pred: Callable[[dict], bool]) -> Callable[[dict], bool]:
	return lambda unit: not pred(unit)


def parse_properties(props):
	return dict(prop.split('=', 1) for prop in props.split(','))


def closer(fh: IO):
	"""Generator that closes fh once it it their turn."""
	if fh.close:
		fh.close()
	yield from []


def is_gzipped(fh: BufferedBinaryIO):
	"""Test if stream is probably a gzip stream"""
	return fh.peek(2).startswith(b'\x1f\x8b')


def make_reader(fh: BufferedBinaryIO, args: Namespace) -> Iterator[dict]:
	pprint(fh)

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


def peek_first_line(fh: BufferedBinaryIO, length=128) -> bytes:
	"""Tries to get the first full line in a buffer that supports peek."""
	while True:
		buf = fh.peek(length)

		pos = buf.find(b'\n')
		if pos != -1:
			return buf[0:pos]

		if len(buf) < len(length):
			return buf

		buf *= 2


def autodetect(fh: BufferedBinaryIO) -> Tuple[str, dict]:
	"""Fill in arguments based on what we can infer from the input we're going
	to get. fh needs to have a peek() method and return bytes."""

	# First test: is it XML?
	xml_signature = b'<?xml '
	if fh.peek(len(xml_signature)).startswith(xml_signature):
		return 'tmx', {}
	
	# Second test: Might it be tab-separated content? And if so, how many columns?
	column_count = peek_first_line(fh).count(b'\t')
	if column_count >= 7:
		return 'tab', {'columns': ['source-document-1', 'source-document-2', 'text-1', 'text-2', 'hash-bifixer', 'score-bifixer', 'score-bicleaner']}

	if column_count >= 5:
		return 'tab', {'columns': ['source-document-1', 'source-document-2', 'text-1', 'text-2', 'score-aligner']}

	raise ValueError('Did not recognize file format')


def autodetect_deduplicator(args, reader):
	"""
	Make a deduplicate filter based on the input options. Fancy bifixer based
	deduplicator if we have the data, otherwise fall back to boring deduplicator.
	"""

	# Grab the first object from the reader to see what we're dealing with
	peeked_obj = next(reader)

	# Stick the peeked object back on :P
	reader = chain([peeked_obj], reader)

	if 'hash-bifixer' in peeked_obj and 'score-bifixer' in peeked_obj:
		return deduplicate(reader,
			key=itemgetter('hash-bifixer'),
			compare=lambda best, new: best['score-bifixer'] < new['score-bifixer'])
	else:
		return deduplicate(reader, key=text_key)


def abort(message):
	"""Abandon ship! Use in case of misguided users."""
	print(message, file=sys.stderr)
	return 1


def main(args, stdin, stdout) -> int:
	parser = ArgumentParser(description='Annotate, filter and convert tmx files')
	parser.add_argument('-i', '--input-format', choices=['tmx', 'tab'])
	parser.add_argument('-o', '--output-format', choices=['tmx', 'tab', 'txt', 'py'], default='tmx')
	parser.add_argument('-l', '--input-languages', nargs=2)
	parser.add_argument('-c', '--input-columns', nargs='+')
	parser.add_argument('--output-languages', nargs='+')
	parser.add_argument('-p', '--properties', action='append', help='List of'
		' A=B,C=D properties to add to each sentence pair. You can use one'
		' --properties for all files or one for each input file.')
	parser.add_argument('-d', '--deduplicate', action='store_true')
	parser.add_argument('--ipc', dest='ipc_meta_files', action='append', type=FileType('r'))
	parser.add_argument('--ipc-group', dest='ipc_group_files', action='append', type=FileType('r'))
	parser.add_argument('--with-bicleaner-score', type=float)
	parser.add_argument('--with-ipc', nargs='+')
	parser.add_argument('--with-ipc-group', nargs='+')
	parser.add_argument('--with-text', nargs='+')
	parser.add_argument('--without-text', nargs='+')
	parser.add_argument('--with-source-document', nargs='+')
	parser.add_argument('--without-source-document', nargs='+')
	parser.add_argument('files', nargs='*', default=[stdin.buffer], type=FileType('rb'))

	# I prefer the modern behaviour where you can do `tmxutil.py -p a=1 file.tmx
	# -p a=2 file2.tmx` etc. but that's only available since Python 3.7.
	if hasattr(parser, 'parse_intermixed_args'):
		args = parser.parse_intermixed_args(args)
	else:
		args = parser.parse_args(args)

	fout = stdout

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
		readers = [({**properties, **unit} for unit in reader) for properties, reader in zip(properties_per_file, readers)]

	# Merge all readers into a single source of sentence pairs
	reader = chain.from_iterable(readers)

	if args.properties and len(args.properties) == 1:
		properties = parse_properties(args.properties[0])
		reader = ({**properties, **unit} for unit in reader)

	# Create writer
	if args.output_format == 'tmx':
		writer = TMXWriter(fout) # type: Writer
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

	# Optional filter & annotation steps for reader
	if args.ipc_meta_files:
		reader = map(IPCLabeler(args.ipc_meta_files).annotate, reader)

	if args.ipc_group_files:
		reader = map(IPCGroupLabeler(args.ipc_group_files).annotate, reader)

	if args.with_bicleaner_score:
		reader = filter(lambda unit: float(unit['score-bicleaner']) >= args.with_bicleaner_score, reader)

	if args.deduplicate:
		reader = autodetect_deduplicator(args, reader)

	if args.with_ipc:
		reader = filter(pred_translation_prop_intersection('ipc', set(args.with_ipc)), reader)
		
	if args.with_ipc_group:
		reader = filter(pred_translation_prop_intersection('ipc-group', set(args.with_ipc_group)), reader)

	if args.with_text:
		reader = filter(pred_translation_text_contains(args.with_text), reader)

	if args.without_text:
		reader = filter(pred_negate(pred_translation_text_contains(args.without_text)), reader)

	if args.with_source_document:
		reader = filter(pred_translation_prop_intersection('source-document', set(args.with_source_document)), reader)

	if args.without_source_document:
		reader = filter(pred_negate(pred_translation_prop_intersection('source-document', set(args.without_source_document))), reader)

	# Main loop. with statement for writer so it can write header & footer
	with writer:
		for unit in reader:
			writer.write(unit)

	return 0


if __name__ == '__main__':
	sys.exit(main(sys.argv[1:], sys.stdin, sys.stdout))
