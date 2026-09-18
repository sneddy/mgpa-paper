# Recorded wet/dry SSVEP

Question: does correction suppress measurement provenance on genuinely
recorded sessions while preserving the twelve-frequency task?

Dry=0 and wet=1 are separate recordings, not simultaneous physiological pairs.
Rotation 1 with split seed 20260902 has 41/20/20/21 participants in
FIT/VAL/HEAD/EVAL (9840/4800/4800/5040 recordings). EEGPT stays frozen. Each
recording is a 15×512 token array, never fifteen independent training examples.

## Corrections and calibration

Closed-form and Iterative MGPA fit on token means and broadcast one recording-specific
displacement to every token. Adapter pooling accumulates in float64 and stores
float32; the independent mean-source reader retains its original float32 mean.
The gate uses six FIT-local virtual transforms: spectral, spatial, and combined
at strengths 0.5/1, estimated from FIT prestimulus data. These are same-trial
synthetic contrasts, not physiological wet/dry pairs. Contrasts are oriented
wet-like to dry-like for both the gate and Closed-form direction. The gate
retains 99% of normalized contrast energy (rank 328).

Published comparisons: Identity, LEACE, both CORAL routes, Closed-form MGPA,
Iterative MGPA, linear-only MGPA, and IGBP. IGBP uses the published longer-training
preset and fits 512-dimensional token means
without using calibration and broadcasts its displacement through exactly the
same interface. FEATMAP is intentionally absent: its physiological pairs do
not exist in these sessions. CORAL alone needs source IDs at application.

## Evaluation

Primary source accessibility uses all 41 readers: 11 mean, 20 direct full-token,
and 10 residual-only readers. HEAD fits the readers and EVAL only scores them.
The invariant centered-token residual bank is fitted once and replayed across
common-offset outputs. The twelve-way token task head receives the full
sequence; frozen and refitted variants are evaluated separately at epoch 50.
Native outputs are all 48 MGPA stages, the full Closed-form step, or 100 IGBP projections.
No endpoint is selected by downstream or EVAL performance.

## Running

Install the package from the repository root. Run the following commands from
this experiment directory:

```sh
python run.py plan
python run.py all --paths configs/paths.local.toml --run-id reproduction_v1
```

Create `paths.local.toml` from `configs/paths.example.toml` and supply verified
prepared data. Alternatively use `configs/raw.example.toml` for raw EEG and a
separately acquired EEGPT checkpoint where required. Raw features are built
before FIT-only coordinates and gates. They may differ in low bits across
encoder backends; freshly generated receipts record this rather than claiming
bitwise equality to a published feature cache.

Phases can run separately: `prepare`, `fit`, `evaluate`, `report`, `verify`.
Only preparation needs `--paths`; subsequent phases use the sealed run inputs.
All declared fits finish and are sealed before the first HEAD/EVAL evaluation.
Fit seeds 17/29/43 are training repetitions on fixed participant roles, not new
subjects. Deterministic maps are fitted once and their reuse is explicit.

Fresh outputs live under `artifacts/runs/<run-id>/`: portable prepared inputs,
numeric model arrays, raw prediction-bearing cells, tables, and short progress
logs. Scientific results are immutable; a changed recipe needs a new run ID.
`verify` replays maps and hashes without retraining readers; it is an output
integrity check, not a second implementation of the complete experiment.

For the shipped numerical results, use `python run.py report --published`.
This regenerates summaries without data/model fitting. It is separate from
`python run.py report --run-id reproduction_v1`, which summarizes a fresh run.
The central paper-asset builder supplies the manuscript layouts.

Source accessibility maximizes participant-mean AUROC over the declared reader
bank within each fit, then averages fits. Task scores are twelve-class macro
AUROC, averaged over cross-source transfer directions. Shared 20,000-draw
participant bootstrap intervals (seed 1101) recompute source maxima and remain
conditional on the fitted maps/readers. The preserved token residuals impose
an accessibility floor; attenuation is not complete source erasure.
