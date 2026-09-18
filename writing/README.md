# MGPA manuscript — merged_v5

This directory contains the self-contained, editable LaTeX project for
*Measurement-Gated Provenance Attenuation for Frozen EEG Representations*.

- Main source: [merged_v5.tex](merged_v5.tex)
- Compiled manuscript: [merged_v5.pdf](merged_v5.pdf)

The manuscript was imported into `sneddy/mgpa-paper` on 2026-09-18. See the
[repository README](../README.md) for the matching implementation, experiment
protocols, and saved-result reproduction guide. Adding these manuscript sources
does not change the released code or numerical artifacts. Historical research
paths mentioned in the manuscript are not extra LaTeX compilation dependencies.

## Compile

From the repository root, with Tectonic installed:

```sh
cd writing
tectonic --keep-logs merged_v5.tex
```

Tectonic handles BibTeX and repeated LaTeX passes. With a full TeX Live/MacTeX
installation, use `latexmk -xelatex merged_v5.tex` from this directory instead.
To edit on Overleaf, upload this directory's contents, select `merged_v5.tex`
as the main document, and select XeLaTeX. Standard LaTeX packages are supplied
by the TeX installation or Overleaf.

The imported project was tested in an isolated directory with Tectonic 0.17.0:
23 pages, with resolved citations and cross-references. All page renders match
the original PDF. The Overleaf service itself has not been tested.

## Where to edit

- `merged_v5.tex`: abstract, introduction, related work, method and theory,
  discussion, statements, and theoretical appendix.
- `sections/experiments.tex`: main experimental section.
- `sections/appendix_protocol_compact.tex`: experimental protocols.
- `sections/appendix_results_compact.tex`: supplementary results.
- `sections/reproducibility.tex`: reproducibility information.
- `assets/*.tex`: editable table contents.
- `references.bib`: bibliography.
- `assets/`: all referenced figures and tables.
- `iclr2027/`: local conference and bibliography styles.

No datasets, checkpoints, or Python experiment environment are needed to build
the manuscript. Keep the directory structure intact. The tracked PDF only
changes when rebuilt; regenerate it after manuscript changes.

`SOURCE_MANIFEST.json` records the initial import. Its `source` paths identify
historical locations in the research workspace; its `file` paths are local to
this directory. These historical paths are not compilation dependencies.
Only dependency paths were relocated during import; no manuscript prose,
equations, numerical results, or figures were changed.
