import re
from datetime import datetime
from typing import TypeVar, Iterable, Optional, Union

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


A = TypeVar('A')
B = TypeVar('B')
def first(it: Iterable[A], default: Optional[B] = None) -> Optional[Union[A,B]]:
	return next(iter(it), default)


