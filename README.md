# tmxutil
tmxutil.py allows you to add domain groups to your Europat tmx files, or filter on them.

## Installation & Requirements
To install tmxutil.py, just download it from [Github](https://github.com/paracrawl/tmxutil) and place it somewhere where you can reach it from the command line. Besides Python 3.5 or newer, it has no external dependencies.

## Examples

Example tmx file: [DE-EN-2001-Abstract.tmx.gz](https://github.com/paracrawl/tmxutil/files/5062356/DE-EN-2001-Abstract.tmx.gz), ipc domain group file: [ipc-groups.tab](https://github.com/paracrawl/tmxutil/raw/master/ipc-groups.tab)

The provided IPC grouping has the following high-level categories:

| Group | Description |
|---|---|
| I | General / Default |
| II | Computing, Science and Tech (Science, photography, optics, cryptography, communications)	|
| III | Biotechnology and Chemical (food, biotech, nanotech, chemistry) |
| IV | Engineering and Manufacturing (Engines, nuclear physica, agriculture, forestry, aviation) |
| V | Daily life (Household, music, arts, clothing, jewlery, sports and decorating) |

**Filtering by IPC code**: Filter out only sentence pairs that come from patents with a certain IPC codes.

```
gzip -cd DE-EN-2001-Abstract.tmx.gz \
| ./tmxutil.py -o tmx --with-ipc D06M15/59 D06P005/02 \
> selection.tmx
```

**Export selection as tab-separated sentence pairs**: By changing the output format of tmxutil you can get the sentence pairs as plain text separated by tabs.

This option can be combined with data augmentation and filter options, although only the first source document per sentence pair is exported. You'll also have to tell it in what order you want the languages to be exported.

```
gzip -cd DE-EN-2001-Abstract.tmx.gz \
| ./tmxutil.py \
    -o tab \
    --output-languages en de \
    --with-ipc D06M15/59 \
> selection-en-de.tsv
```

**Adding ipc groups to tmx file**: To be able to make more coarse-grained selections you can add ipc groups (c.f. domains) to the sentence pairs, based on the IPC codes already in the tmx file. You can then use those ipc groups to make a selection using `--with-ipc-group`, which works just like `--with-ipc`.

The ipc-groups.tab file used here should have a IPC code prefix and a group name on each line, separated by a tab, as the first two columns. You can get the ipc-groups.tab file from [the project's releases page](https://github.com/paracrawl/tmxutil/releases).

```
gzip -cd DE-EN-2001-Abstract.tmx.gz \
| ./tmxutil.py \
	-o tmx \
	--ipc-group ipc-groups.tab \
| gzip > DE-EN-2001-Abstract-with-groups.tmx.gz
```

Only the tmx output format will maintain the ipc-group metadata by adding ipc-group properties. Other output formats won't maintain it, but you can still use `--with-ipc-group` directly to make a selection.

***Converting tsv to tmx***: tmxutil can also be used to generate tmx files from sentence pairs. The input format is the same as the `tab` output format, that is `source1 \t source2 \t sentence1 \t sentence2`.

To also add the IPC codes from metadata, use the `--ipc` option. The format of this file should be `l1_id \t _ \t _ \t _ \t l1_lang \t l1_ipcs \t l2_id \t _ \t _ \t _ \t l2_lang \t l2_ipcs` where `id` is the document identifier, and `l1_ipc` is a comma-separated list of all ipc codes for this document.

```
cat DE-EN-2001-Abstract-aligned.tsv \
| ./tab2tmx.py \
    -o tmx \
    -l de en \
    -d \
    --ipc DE-EN-2001-Metadata.tab \
| gzip -9c > DE-EN-2001-Abstract.tmx.gz
```

## Parameters
- `-i tmx|tab, --input-format tmx|tab` input format, if not given will be auto-detected. Possible values: `tmx`, `tab`.
	- In case of `tab` you'll have to specify which languages are in there using `--languages l1 l2`.
- `-o tmx|tab|txt, --output-format tmx|tab|txt` output format, either `tmx`, `tab` or `txt`.
	- In case of `tab` you'll have to specify the languages, e.g. `--output-languages l1 l2`.
	- When using `txt`, you'll have to select which language you want the plain text for, i.e. `--output-languages en`.
- `-l L1 L2, --input-languages L1 L2`. Languages & order of them in the input file. Only necessary when reading `tab` files.
- `--ouput-languages L1 [L2]` language or order of languages in the output file. Not used if `tmx` is the output.
- `-d, --deduplicate` groups sentence pairs with the same text or hash together.
- `--drop PROP [PROP ...]` drop properties from the sentence pairs while writing output.
- `--renumber-output` causes all translation unit ids to be reset. Enabled by default when multiple input files are given.
- `--ipc FILE` adds IPC metadata to each sentence pair.
- `--with PROP=VALUE [PROP=VALUE ...]` filters sentence pairs on their text or properties. Supported operators are `=`, `>`, `<`, `>=`, `<=` and `=~` for regular expressions. Use multiple `PROP=VALUE` pairs in a `--with` option to combine the conditions (i.e. AND). Or use multiple `--with` options for separate conditions (i.e. OR).
- `--without PROP=VALUE [PROP=VALUE ...]` same as `--with`, but negated, for excluding instead of including sentence pairs. 
- `--verbose` enabled progress updates.
