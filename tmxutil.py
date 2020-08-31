#!/usr/bin/env python3
# Tool to convert between tab, txt and tmx formatting, filtering & adding
# annotations.

__VERSION__ = 1.0

import csv
import sys
import time
import re
from abc import ABC, abstractmethod
from contextlib import contextmanager
from io import BufferedReader
from xml.sax.saxutils import escape, quoteattr
from xml.etree.ElementTree import iterparse
from itertools import combinations
from collections import defaultdict
from pprint import pprint
from typing import Callable, List, Optional, Any, Iterator, Set
from operator import itemgetter

class XMLWriter(object):
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
	def __iter__(self) -> Iterator[dict]:
		return self.records()

	@abstractmethod
	def records(self) -> Iterator[dict]:
		pass


class Writer(ABC):
	def __enter__(self):
		return self

	def __exit__(self, *args):
		pass

	@abstractmethod
	def write(self, unit: dict):
		pass


class TMXReader(Reader):
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

		unit = None
		translation = None

		lang_key = '{http://www.w3.org/XML/1998/namespace}lang'
		
		for event, element in iterparse(self.fh, events={'start', 'end'}):
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
						if key != 'text':
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
	def __init__(self, paths: List[str] = []):
		self.lut = dict()
		for path in paths:
			with open(path, 'r') as fh:
				self.load(fh)

	def load(self, fh):
		for line in fh:
			src_id, _, _, _, src_lang, src_ipcs, trg_id, _, _, _, trg_lang, trg_ipcs = line.split('\t', 11)
			self.lut[(src_lang.lower(), src_id)] = set(ipc.strip() for ipc in src_ipcs.split(','))
			self.lut[(trg_lang.lower(), trg_id)] = set(ipc.strip() for ipc in trg_ipcs.split(','))

	def annotate(self, unit: dict) -> dict:
		for lang, translation in unit['translations'].items():
			keys = self.lut.keys() & {(lang.lower(), url) for url in translation['source-document']}
			translation['ipc'] = set().union(*(self.lut[key] for key in keys))
		return unit


class IPCGroupLabeler(object):
	def __init__(self, paths: List[str] = []):
		self.patterns = []
		for path in paths:
			with open(path, 'r') as fh:
				self.load(fh)

	def load(self, fh):
		for line in fh:
			prefix, group, *_ = line.split('\t', 2)
			self.patterns.append((
				prefix.strip(),
				{prefix.strip(), group.strip()} if prefix.strip() != "" else {group.strip()}
			))

		# Sort with most specific on top
		self.patterns.sort(key=lambda pattern: (-len(pattern[0]), pattern[0]))

	def find_group(self, ipc_code: str) -> Optional[str]:
		for prefix, groups in self.patterns:
			if ipc_code.startswith(prefix):
				return groups
		return set()

	def annotate(self, unit: dict) -> dict:
		for lang, translation in unit['translations'].items():
			translation['ipc-group'] = set().union(*map(self.find_group, translation['ipc']))
		return unit


def text_key(unit: dict) -> tuple:
	return tuple(translation['text'] for translation in unit['translations'].values())


def deduplicate(reader: Iterator[dict], key: Callable[[dict], Any], compare: Callable[[dict, dict], bool] = lambda _: False) -> Iterator[dict]:
	best = dict()

	for unit in reader:
		unit_id = key(unit)

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


def peek_first_line(fh, length=128) -> bytes:
	"""Tries to get the first full line in a buffer that supports peek"""
	while True:
		buf = fh.peek(length)

		pos = buf.find(b'\n')
		if pos != -1:
			return buf[0:pos]

		if len(buf) < len(length):
			return buf

		buf *= 2


def autodetect(args, fh) -> Optional[str]:
	"""Fill in arguments based on what we can infer from the input we're going to get."""

	# First test: is it XML?
	xml_signature = b'<?xml '
	if fh.buffer.peek(len(xml_signature)).startswith(xml_signature):
		args.input_format = 'xml'
		return
	
	# Second test: Might it be tab-separated content? And if so, how many columns?
	column_count = peek_first_line(fh.buffer).count(b'\t')
	if column_count >= 7:
		args.input_format = 'tab'
		args.input_columns = ['source-document-1', 'source-document-2', 'text-1', 'text-2', 'hash-bifixer', 'score-bifixer', 'score-bicleaner']
		return

	if column_count >= 5:
		args.input_format == 'tab'
		args.input_columns =['source-document-1', 'source-document-2', 'text-1', 'text-2', 'score-aligner']
		return


def autodetect_deduplicator(args, reader):
	"""
	Make a deduplicate filter based on the input options. Fancy bifixer based
	deduplicator if we have the data, otherwise fall back to boring deduplicator.
	"""
	if args.input_format == 'tab' and 'hash-bifixer' in args.input_columns and 'score-bifixer' in args.input_columns:
		return deduplicate(reader,
			key=itemgetter('hash-bifixer'),
			compare=lambda best, new: best['score-bifixer'] < new['score-bifixer'])
	else:
		return deduplicate(reader, key=text_key)


def abort(message):
	print(message, file=sys.stderr)
	sys.exit(1)


if __name__ == '__main__':
	from argparse import ArgumentParser

	parser = ArgumentParser(description='Annotate, filter and convert tmx files')
	parser.add_argument('-i', '--input-format', choices=['tmx', 'tab'])
	parser.add_argument('-o', '--output-format', choices=['tmx', 'tab', 'txt', 'py'], required=True)
	parser.add_argument('-l', '--input-languages', nargs=2)
	parser.add_argument('-c', '--input-columns', nargs='+')
	parser.add_argument('--output-languages', nargs='+')
	parser.add_argument('-d', '--deduplicate', action='store_true')
	parser.add_argument('--ipc', dest='ipc_meta_files', action='append')
	parser.add_argument('--ipc-group', dest='ipc_group_files', action='append')
	parser.add_argument('--with-ipc', nargs='+')
	parser.add_argument('--with-ipc-group', nargs='+')
	parser.add_argument('--with-text', nargs='+')
	parser.add_argument('--without-text', nargs='+')
	parser.add_argument('--with-source-document', nargs='+')
	parser.add_argument('--without-source-document', nargs='+')

	args = parser.parse_args()

	fin = sys.stdin
	fout = sys.stdout

	reader = None
	writer = None

	# Autodetect input format if we're lazy
	if not args.input_format:
		autodetect(args, fin)

	# Parameter validation, early warning system
	if not args.input_format:
		abort("Use --input-format to specify the input format")

	if args.input_format == 'tab' and not args.input_languages:
		abort("Tab input format requires a --input-languages LANG1 LANG2 option.")

	if args.input_format == 'tab' and not args.input_columns:
		abort("Tab input format requires a --input-columns option.")

	if args.output_languages == 'txt' and (not args.output_languages or len(args.output_languages) != 1):
		abort("Use --output-languages X to select which language. When writing txt, it can only write one language at a time.")

	# Create reader
	if args.input_format == 'tmx':
		reader = TMXReader(fin)
	elif args.input_format == 'tab':
		reader = TabReader(fin, *args.input_languages, columns=args.input_columns)
	else:
		raise RuntimeError("Could not create input reader")

	# Create writer
	if args.output_format == 'tmx':
		writer = TMXWriter(fout)
	elif args.output_format == 'tab':
		writer = TabWriter(fout, args.output_languages)
	elif args.output_format == 'txt':
		writer = TxtWriter(fout, args.output_languages[0])
	elif args.output_format == 'py':
		writer = PyWriter(fout)

	# Optional filter & annotation steps for reader
	if args.ipc_meta_files:
		reader = map(IPCLabeler(args.ipc_meta_files).annotate, reader)

	if args.ipc_group_files:
		reader = map(IPCGroupLabeler(args.ipc_group_files).annotate, reader)

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