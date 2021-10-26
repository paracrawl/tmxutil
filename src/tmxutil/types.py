from abc import ABCMeta, ABC, abstractmethod
from typing import Optional, Type, Iterator, Dict, Set, BinaryIO
from types import TracebackType


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

