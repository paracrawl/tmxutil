__version__ = 'dev'

try:
	import pkg_resources
	__version__ = pkg_resources.require('tmxutil-pkg-jelmervdl')[0].version
except:
	pass
