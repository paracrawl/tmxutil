from tmxutil.types import Writer, TranslationUnit
from tmxutil.interactive import tqdm
from collections import Counter
from time import time
from typing import Optional, Callable, TextIO, Iterable, Any, Type
from types import TracebackType
from operator import itemgetter


class CountWriter(Writer):
	"""Instead of writing tmx records, it counts a property and writes a summary
	of which values it encountered for that property, and how often it encountered
	them."""
	def __init__(self, fh: TextIO, key: Callable[[TranslationUnit], Iterable[Any]]):
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
	def __init__(self, fh: TextIO, key: Callable[[TranslationUnit], Iterable[Any]]):
		self.fh = fh
		self.key = key
		self.top_n = 10

	def __enter__(self) -> 'LiveCountWriter':
		self.counter = Counter() # type: Counter[Any]
		self.total = 0
		self.bars: tqdm = []
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
