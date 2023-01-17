#!/usr/bin/env python3
# Tool to convert between tab, txt and tmx formatting, filtering & adding
# annotations.
import re
import operator
import sys
import os
import importlib
from contextlib import ExitStack
from datetime import datetime
from textwrap import dedent
from argparse import ArgumentParser, FileType, Namespace, RawDescriptionHelpFormatter
from logging import info, getLogger, INFO, ERROR
from typing import cast, Callable, Tuple, Iterator, Iterable, Any, Type, TypeVar, List, Dict, Set, Mapping, Optional
from functools import reduce, partial
from io import TextIOWrapper
from itertools import chain, starmap
from . import __version__
from .types import Reader, Writer, TranslationUnit, BufferedBinaryIO
from .utils import first, fromisoformat, fromfilesize
from .filters.deduplicate import deduplicate
from .filters.ipc import IPCLabeler, IPCGroupLabeler
from .formats import make_reader
from .formats.count import CountWriter, LiveCountWriter
from .formats.json import JSONWriter
from .formats.pickle import PickleReader, PickleWriter
from .formats.tab import TabReader, TabWriter
from .formats.tmx import TMXReader, TMXWriter
from .formats.txt import TxtWriter
from .interactive import tqdm


def text_key(unit: TranslationUnit) -> Tuple[str,...]:
	return tuple(translation.text for translation in unit.translations.values())


T = TypeVar('T', float, str)

def build_binary_condition(type: Type[T], op: Callable[[T,T], bool]) -> Callable[[Callable[[TranslationUnit], Iterable[Any]],str], Callable[[TranslationUnit], bool]]:
	"""Wrapper for standard python operations on types. I.e. to implement gt
	and lt."""
	def build_condition(lhs: Callable[[TranslationUnit], Iterable[Any]], rhs: str) -> Callable[[TranslationUnit], bool]:
		return lambda unit: any(op(type(el), type(rhs)) for el in lhs(unit))
	return build_condition


def build_regex_condition(lhs: Callable[[TranslationUnit], Iterable[Any]], rhs: str) -> Callable[[TranslationUnit], bool]:
	"""Specialised version (or wrapper around) build_binary_condition that makes
	one that tests a regular expression."""
	pattern = re.compile(rhs)
	return lambda unit: any(pattern.search(str(el)) is not None for el in lhs(unit))


condition_operators = {
	 '<': build_binary_condition(float, operator.lt),
	 '>': build_binary_condition(float, operator.gt),
	'<=': build_binary_condition(float, operator.le),
	'>=': build_binary_condition(float, operator.ge),
	 '=': build_binary_condition(str, operator.eq),
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


def parse_condition(operators: Mapping[str,Callable[[str,str], Callable[[TranslationUnit], bool]]], expr: str, functions={}) -> Callable[[TranslationUnit], bool]:
	pattern = r'^(?P<lhs>.+?)(?P<op>{operators})(?P<rhs>.*)$'.format(
		operators='|'.join(re.escape(op) for op in sorted(operators.keys(), key=len, reverse=True)))

	match = re.match(pattern, expr)
	
	if match is None:
		raise ValueError("Could not parse condition '{}'".format(expr))

	info("Using expression op:'%(op)s' lhs:'%(lhs)s' rhs:'%(rhs)s'", match.groupdict())

	prop_getter = parse_property_getter(match.group('lhs'), functions=functions)

	return operators[match.group('op')](prop_getter, match.group('rhs'))


def parse_property_getter(expr: str, functions: Mapping[str,Callable[[Any],Any]] = {'len': len}) -> Callable[[TranslationUnit], Iterable[Any]]:
	ops = [] #type: List[Callable[[Any], Any]]

	while True:
		match = re.match(r'^(?P<fun>[a-zA-Z_]\w*)\((?P<expr>.+?)\)$', expr)
		if not match:
			break

		if not match.group('fun') in functions:
			raise ValueError('Function `{}` in expression `{}` not found.'.format(match.group('fun'), expr))

		ops.insert(0, functions[match.group('fun')])
		expr = match.group('expr')

	match = re.match(r'^((?P<lang>[\w-]+)?(?P<dot>\.))?(?P<prop>[\w-]+)(?P<brackets>\[\])?$', expr)
	if not match:
		raise ValueError('Could not interpret expression `{}`'.format(expr))

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


def main(argv: List[str], stdin: BufferedBinaryIO, stdout: BufferedBinaryIO) -> int:
	parser = ArgumentParser(
		formatter_class=RawDescriptionHelpFormatter,
		description='Annotate, analyze, filter and convert (mainly) tmx files',
		epilog=dedent('''
		Supported syntax for FILTER_EXPR:
		  Syntax: PROP_EXPR OPERATOR VALUE where:
		    PROP_EXPR           Either 'text' or the value of the "type" attribute
		                        of a <prop/> element.
		    OPERATOR            Supported operators:
		                          >, >=, <, <= for numeric comparisons.
		                          =, =~ for string comparisons.
		    VALUE               String, number or regular expression.

		  Examples:
		    collection=europat  Matches sentence pairs that have a property 
		                        'collection' that is exactly 'europat'.
		    text=~euro.*        Matches pairs that match a regular expression.
		    id>400              Matches pairs that have an id larger than 400

		Supported syntax for PROP_EXPR:
		  Syntax: [FUNCTION] ( [LANG] [.] PROPERTY [\\[\\]] ) where all except
		          PROPERTY is optional. If FUNCTION is not used, you don't need the
		          parenthesis. The [] after PROPERTY can be used to indicate that
		          all values of that property for a <tu/> should be treated as a
		          single set.

		  Examples:
		    source-document     Count every prop type "source-document", either as
		                        part of the <tu/> or <tuv/>.
		    .collection         Count every collection observed as a prop of the
		                        sentence pair.
		    .collection[]       Count every combination of <prop type="collection"/>
		                        observed in <tu/>.
		    len(en.text)        String length of english side of the sentence pair.
		                        You can use your own functions using --include.
	'''))
	parser.add_argument('--version', action='version', version=__version__)
	parser.add_argument('-i', '--input-format', choices=['tmx', 'tab', 'pickle'], help='Input file format. Automatically detected if left unspecified.')
	parser.add_argument('-o', '--output-format', choices=['tmx', 'tab', 'txt', 'json', 'pickle'], default='tmx', help='Output file format. Output is always written to stdout.')
	parser.add_argument('-l', '--input-languages', nargs=2, help='Input languages in case of tab input. Needs to be in order their appearance in the columns.')
	parser.add_argument('-c', '--input-columns', nargs='+', help='Input columns in case of tab input. Column names ending in -1 or -2 will be treated as translation-specific.')
	parser.add_argument('--output-languages', nargs='+', help='Output languages for tab and txt output. txt output allows only one language, tab multiple.')
	parser.add_argument('--output-columns', metavar="PROP_EXPR", nargs='+', help='Output columns for tab output. Use {lang}.{property} syntax to select language specific properties such as en.source-document or de.text.')
	parser.add_argument('--output', default=stdout, type=FileType('wb'), help='Output file. Defaults to stdout.')
	parser.add_argument('--creation-date', type=fromisoformat, default=datetime.now(), help='override creation date in tmx output.')
	parser.add_argument('-p', '--properties', action='append', help='List of A=B,C=D properties to add to each sentence pair. You can use one --properties for all files or one for each input file.')
	parser.add_argument('-d', '--deduplicate', action='store_true', help='Deduplicate units before printing. Unit properties are combined where possible. If score-bifixer and hash-bifixer are avaiable, these will be used.')
	parser.add_argument('--drop', nargs='+', dest='drop_properties', help='Drop properties from output.')
	parser.add_argument('--renumber-output', action='store_true', help='Renumber the translation unit ids. Always enabled when multiple input files are given.')
	parser.add_argument('--ipc', dest='ipc_meta_files', nargs="+", type=FileType('r'), help='One or more IPC metadata files.')
	parser.add_argument('--ipc-group', dest='ipc_group_files', nargs="+", type=FileType('r'), help='One or more IPC grouping files.')
	parser.add_argument('--with', nargs='+', action='append', dest='filter_with', metavar='FILTER_EXPR')
	parser.add_argument('--without', nargs='+', action='append', dest='filter_without', metavar='FILTER_EXPR')
	parser.add_argument('-P', '--progress', action='store_true', help='Show progress bar when reading files.')
	logging_options = parser.add_mutually_exclusive_group()
	logging_options.add_argument('-q', '--quiet', action='store_true', help='Hide issues encountered while reading files.')
	logging_options.add_argument('-v', '--verbose', action='store_true', help='Print progress updates.')
	parser.add_argument('--workspace', type=fromfilesize, help='Mamimum memory usage for deduplication. When exceeded, will continue deduplication using filesystem.', default='4G')
	parser.add_argument('--count', dest='count_property', help='Count which values occur for a property.', metavar='COUNT_EXPR')
	parser.add_argument('--include', action='append', default=[], dest='count_libraries', help='Include a python file so functions defined in that file can be used with --count, e.g. include something that provides a domain(url:str) function, and use `--count domain(source-document)`.')
	parser.add_argument('files', nargs='*', default=[stdin], type=FileType('rb'), help='Input files. May be gzipped. If not specified stdin is used.')

	# I prefer the modern behaviour where you can do `tmxutil.py -p a=1 file.tmx
	# -p a=2 file2.tmx` etc. but that's only available since Python 3.7.
	if hasattr(parser, 'parse_intermixed_args'):
		args = parser.parse_intermixed_args(argv)
	else:
		args = parser.parse_args(argv)

	if args.verbose:
		getLogger().setLevel(INFO)
	elif args.quiet:
		getLogger().setLevel(ERROR)

	# Load in functions early so if anything is wrong with them we'll know before
	# we attempt to parse anything.
	functions = reduce(lambda obj, file: {**obj, **import_file_as_module(file).__dict__},
	                   args.count_libraries, {'len': len})

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
		dnf = [[parse_condition(condition_operators, cond_str, functions=functions) for cond_str in cond_expr] for cond_expr in args.filter_with]
		reader = filter(lambda unit: any(all(expr(unit) for expr in cond) for cond in dnf), reader)

	if args.filter_without:
		dnf = [[parse_condition(condition_operators, cond_str, functions=functions) for cond_str in cond_expr] for cond_expr in args.filter_without]
		reader = filter(lambda unit: all(any(not expr(unit) for expr in cond) for cond in dnf), reader)

	if args.deduplicate:
		reader = make_deduplicator(args, reader, mem_limit=args.workspace)

	if args.renumber_output:
		reader = starmap(partial(set_property, 'id'), enumerate(reader, start=1))

	# If we want to drop properties from the output, do that as the last step.
	if args.drop_properties:
		reader = map(partial(del_properties, args.drop_properties), reader)

	# Create writer
	with ExitStack() as ctx:
		if args.output_format == 'pickle':
			writer = ctx.enter_context(PickleWriter(args.output)) # type: Writer

		else:
			text_out = ctx.enter_context(TextIOWrapper(args.output, encoding='utf-8'))
			
			if args.count_property:
				count_property = parse_property_getter(args.count_property, functions=functions)
					

				if tqdm and args.progress:
					writer = ctx.enter_context(LiveCountWriter(text_out, key=count_property))
				else:
					writer = ctx.enter_context(CountWriter(text_out, key=count_property))
			elif args.output_format == 'tmx':
				writer = ctx.enter_context(TMXWriter(text_out, creation_date=args.creation_date))
			elif args.output_format == 'tab':
				if not args.output_columns:
					if not args.output_languages:
						return abort("Use --output-languages X Y to select the order of the columns in the output, or use --output-columns directly.")
					args.output_columns = [
						*(f'{lang}.source-document' for lang in args.output_languages),
						*(f'{lang}.text' for lang in args.output_languages)
					]

				column_getters = [
					parse_property_getter(expr, functions=functions)
					for expr in args.output_columns
				]

				writer = ctx.enter_context(TabWriter(text_out, column_getters))
			elif args.output_format == 'txt':
				if not args.output_languages or len(args.output_languages) != 1:
					return abort("Use --output-languages X to select which language."
					             " When writing txt, it can only write one language at"
					             " a time.")
				writer = ctx.enter_context(TxtWriter(text_out, args.output_languages[0]))
			elif args.output_format == 'json':
				writer = ctx.enter_context(JSONWriter(text_out))
			elif args.output_format == 'pickle':
				writer = ctx.enter_context(PickleWriter(args.output))
			else:
				raise ValueError('Unknown output format: {}'.format(args.output_format))

		# Main loop. with statement for writer so it can write header & footer
		count = 0
		for unit in reader:
			writer.write(unit)
			count += 1
		info("Written %d records.", count)

	return 0


def entrypoint():
	"""main() but with all the standard parameters passed in"""
	try:
		sys.exit(main(sys.argv[1:],
			cast(BufferedBinaryIO, sys.stdin.buffer),
			cast(BufferedBinaryIO, sys.stdout.buffer)))
	except ValueError as e:
		sys.exit(abort("Error: {}".format(e)))
