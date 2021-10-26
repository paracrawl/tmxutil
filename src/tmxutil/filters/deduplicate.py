import pickle
import resource
from typing import Iterator, Callable, Any, Dict, List
from itertools import chain
from logging import info
from collections import OrderedDict
from tempfile import TemporaryFile
from ..types import TranslationUnit


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