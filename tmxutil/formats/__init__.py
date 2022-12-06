import gzip
from io import TextIOWrapper
from typing import Any, Generator, Optional, Iterable, Sequence, cast, Iterator, Tuple, Dict
from itertools import chain
from ..types import Reader, TranslationUnit, BufferedBinaryIO
from ..interactive import tqdm, ProgressWrapper
from .pickle import PickleReader
from .tmx import TMXReader
from .tab import TabReader

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

