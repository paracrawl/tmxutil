import sys
import os
from typing import Any


try:
	from tqdm.autonotebook import tqdm
except ImportError:
	tqdm = None

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
