import tmxutil
from tmxutil.types import Reader, Writer, TranslationUnit, TranslationUnitVariant
from tmxutil.utils import first
from typing import BinaryIO, Iterator, List, Mapping, Set, Tuple, TextIO, Optional, Type
from types import TracebackType
from logging import info, warning
from xml.etree.ElementTree import iterparse, Element
from datetime import datetime


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


class TMXWriter(Writer):
	def __init__(self, fh: TextIO, *, creation_date: Optional[datetime] = None):
		self.fh = fh
		self.creation_date = creation_date or datetime.now()
		
	def __enter__(self) -> 'TMXWriter':
		self.fh.write('<?xml version="1.0"?>\n'
					  '<tmx version="1.4">\n'
					  '  <header\n'
					  '    o-tmf="PlainText"\n'
					  '    creationtool="tmxutil"\n'
					  '    creationtoolversion="' + tmxutil.__version__ + '"\n'
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
			if key != 'id' and value:
				self.fh.write('      <prop type="' + _escape_attrib(str(key)) + '">' + _escape_cdata(str(value)) + '</prop>\n')
		
		for lang, translation in sorted(unit.translations.items()):
			self.fh.write('      <tuv xml:lang="' + _escape_attrib(lang) + '">\n')
			for key, value in sorted(_flatten(translation)):
				if value:
					self.fh.write('        <prop type="' + _escape_attrib(str(key)) + '">' + _escape_cdata(str(value)) + '</prop>\n')
			self.fh.write('        <seg>' + _escape_cdata(str(translation.text)) + '</seg>\n'
			              '      </tuv>\n')
		self.fh.write('    </tu>\n')

