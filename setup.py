import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
	long_description = fh.read()

setuptools.setup(
	name="tmxutil-pkg-jelmervdl",
	version="1.2",
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
		'console_scripts': [
			'tmxutil=tmxutil.cli:entrypoint',
		],
	},
	package_dir={"": "src"},
	packages=setuptools.find_packages(where="src"),
	python_requires=">=3.6",
)