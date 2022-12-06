import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
	long_description = fh.read()

setuptools.setup(
	name="tmxutil",
	version="1.3",
	author="Jelmer van der Linde",
	author_email="jelmer@ikhoefgeen.nl",
	description="Tool to create, augment, filter and generally work with TMX files.",
	long_description=long_description,
	long_description_content_type="text/markdown",
	url="https://github.com/paracrawl/tmxutil",
	project_urls={
		"Bug Tracker": "https://github.com/paracrawl/tmxutil/issues",
	},
	classifiers=[
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: MIT License",
		"Operating System :: OS Independent",
	],
	entry_points={
		"console_scripts": [
			"tmxutil=tmxutil.cli:entrypoint",
		],
	},
	packages=[
		"tmxutil",
		"tmxutil.filters",
		"tmxutil.formats"
	],
	python_requires=">=3.6",
)