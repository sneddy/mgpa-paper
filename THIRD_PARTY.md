# Third-party methods and inputs

This release retains the original project's MIT license. It does not redistribute
raw participant recordings or pretrained weights.

The [download guide](DATASETS.md) lists official sources, required files and
local directory layouts for [SSVEP](DATASETS.md#ssvep),
[Flex–Neuroscan](DATASETS.md#flex-neuroscan) and [EEGPT](DATASETS.md#eegpt).

- **LEACE, CORAL and FEATMAP:** local numerical implementations of the comparison
  methods cited in the manuscript. They are baselines, not MGPA components:
  [LEACE (Belrose et al., 2023)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d066d21c619d0a78c5b557fa3291a8f4-Abstract-Conference.html),
  [CORAL (Sun et al., 2016)](https://arxiv.org/abs/1511.05547), and
  [FEATMAP (Donle et al., 2026)](https://doi.org/10.64898/2026.07.02.736184).
- **IGBP:** binary projective correction following Iskander et al., Findings of
  ACL 2023. The implementation follows [the paper](https://aclanthology.org/2023.findings-acl.369/)
  and the authors' repository `technion-cs-nlp/igbp_nonlinear-removal`, inspected
  at commit `a61ea88112665012d9bdb19d3b172b4088e024fa`. The code documents its
  logit-space evaluation and the published longer-training preset. It does not
  use MGPA's gate or conditional anchor.
- **EEGPT:** supplied through the separately installed Braindecode implementation.
  Acquisition requires the explicit checkpoint documented in the experiment
  READMEs. Download and use it under its provider's terms; no weight file is
  included here.
- **Data:** the wet/dry SSVEP and simultaneous Neuroscan/Flex recordings must be
  obtained from their providers. Acquisition commands only read a user-supplied
  data root. They do not transfer or publish participant recordings.
  The SSVEP input is Figshare record `13560281`, with `S001.mat` through
  `S102.mat` and `Subjects_Information.mat`. The simultaneous recording source is
  [OSF project zj3f5](https://osf.io/zj3f5/), “Emotiv EPOC Flex Saline Validation”
  ([study](https://doi.org/10.7717/peerj.9713)); retain P300, N170 and MMN under
  the original `raw/Saline Raw Data/` layout.
- **Dependencies:** NumPy, SciPy, scikit-learn, PyTorch, Matplotlib and the optional
  EEG preprocessing libraries keep their respective licenses. Package installation
  obtains them normally; their code is not vendored into this release.
