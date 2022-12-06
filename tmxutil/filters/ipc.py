from typing import List, TextIO, Dict, Tuple, Set
from tmxutil.types import TranslationUnit
from logging import warning


class IPCLabeler(object):
	"""Add IPC labels to sentence pairs based on the patent ids found in the
	source-document property of either side of the pair."""

	#lut: Dict[Tuple[str,str], Set[str]]

	def __init__(self, files: List[TextIO] = []):
		self.lut = dict() # type: Dict[Tuple[str,str], Set[str]]
		for fh in files:
			self.load(fh)

	def load(self, fh: TextIO) -> None:
		for line in fh:
			parts = line.split('\t', 11)
			if len(parts) != 6 and len(parts) != 12:
				warning("Expected 6 or 12 fields while reading IPC file, found %d, in %s:%d", len(parts), fh.name, line)
				continue
			src_id, _, _, _, src_lang, src_ipcs = parts[:6]
			self.lut[(src_lang.lower(), src_id)] = set(ipc.strip() for ipc in src_ipcs.split(',') if ipc.strip() != '')
			if len(parts) == 12:
				trg_id, _, _, _, trg_lang, trg_ipcs = parts[6:]
				self.lut[(trg_lang.lower(), trg_id)] = set(ipc.strip() for ipc in trg_ipcs.split(',') if ipc.strip() != '')

	def annotate(self, unit: TranslationUnit) -> TranslationUnit:
		for lang, translation in unit.translations.items():
			# Ignoring type because https://github.com/python/mypy/issues/2013
			translation['ipc'] = set().union(*(
				self.lut[(lang.lower(), url)]
				for url in translation['source-document']
				if (lang.lower(), url) in self.lut
			)) # type: ignore
		return unit


class IPCGroupLabeler(object):
	"""Add overall IPC group ids based on IPC labels added by IPCLabeler."""

	#patterns: List[Tuple[str,Set[str]]]

	def __init__(self, files: List[TextIO] = []):
		self.patterns = [] # type: List[Tuple[str,Set[str]]]
		for fh in files:
			self.load(fh)

	def load(self, fh: TextIO) -> None:
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

	def annotate(self, unit: TranslationUnit) -> TranslationUnit:
		for lang, translation in unit.translations.items():
			translation['ipc-group'] = set().union(*map(self.find_group, translation['ipc'])) # type: ignore
		return unit