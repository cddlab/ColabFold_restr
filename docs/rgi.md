# RGI in ColabFold

Open [ColabFold2 preview](https://colab.research.google.com/github/th2ch-g/ColabFold/blob/rgi-integration/ColabFold2_preview.ipynb),
choose a model and a GPU runtime, enter your molecules, and run the cells in order.
Set **use_rgi** to enable restraints. Leave it **off** for vanilla prediction.

| Notebook | RGI models |
| --- | --- |
| ColabFold2 preview | AlphaFold3, OpenFold3, Boltz2, Protenix2, RoseTTAFold3, Chai1, OpenDDE, ESMFold2, ESMFold2 LM600M and LM300M |
| [AlphaFold3 / OpenFold3](https://colab.research.google.com/github/th2ch-g/ColabFold/blob/rgi-integration/AlphaFold3_of3.ipynb) | AlphaFold3 and OpenFold3, using the same maintained runtime |
| [Boltz-1](https://colab.research.google.com/github/th2ch-g/ColabFold/blob/rgi-integration/Boltz1.ipynb) | Native Boltz-1, using the existing RGI predictor fork |

OpenBind0, IntelliFold2 and AlphaFold2 remain available as vanilla models in preview.
Enabling RGI for them raises an actionable error. ESMFold2's language-model input
currently supports one protein chain; choose another predictor for a protein complex.
The preview uses ColabFold's JAX ports, not the original PyTorch implementations.

## First distance restraint

1. Enable **use_rgi** in the install cell. For Boltz-1, this switch is in the input cell.
2. Run the molecule input cell and inspect the printed chain table. Numbering starts
   at residue **1 in each chain**, independently of numbering in a reference structure.
3. Choose **distance**. Enter the two chain IDs and residue ranges, such as chain A,
   `1-10`, and chain A, `40-50`. A comma adds another range: `1-10,25-30`.
4. Choose all heavy atoms, backbone atoms, or C-alpha (`CA`). The distance is measured
   between the groups' geometric centroids, in Angstrom.
5. Set the desired distance. Tolerance 0 targets that value. A positive tolerance
   leaves a free interval around it; for example 25 +/- 2 permits 23 to 27 Angstrom.
6. Run prediction. Inspect the nonzero restraint counts and measured final distances.

The form prints the generated shared configuration before inference. All atom selection
and optimization use RGI-toolkit. Selections that match no atoms fail before the model
forward pass. An enabled configuration that builds no active restraints also fails.

## Ligand geometry and advanced settings

Add a ligand, then choose **ligand_geometry** or **distance+ligand_geometry**.
The default `rgi_conformer_chains=ligands` opts all ligand chains in. To select particular
chains, enter their IDs separated by commas, for example `B,C`. Protein and nucleic-acid
chains are never implicitly opted into geometry restraints. For deliberate polymer
geometry work, consider the shared `monomer_library: true` option.

The preset enables the toolkit's bond, angle, chiral, cis/trans and VdW defaults, plus
planar-group restraints. VdW covers intra- and intermolecular interactions. A monatomic
ion has no internal bonds or angles; read the built counts rather than assuming every
term applies to every molecule.

Choose **custom** to use the full
[RGI-toolkit configuration](https://github.com/cddlab/rgi_toolkit/blob/main/docs/config.md):
distance, angle, dihedral, improper, chiral, plane, RMSD, base pairing, reference-backed
groups and custom energies all use the same config dictionary. Paste YAML/JSON into
`rgi_custom`, or upload a file through Colab's Files panel and enter `rgi_config_path`.
Use one source at a time. Upload any referenced structures as well; paths inside an
external config are relative to that file. Conformer configurations also need the
intended chain IDs in `rgi_conformer_chains`.

```yaml
verbose: true
distance_restraints_config:
  - atom_selection1: chain A and resid 1 to 10
    atom_selection2: chain A and resid 40 to 50
    harmonic: {target_distance: 25.0}
```

RGI guides sampling; it does not establish that a proposed restraint is biologically
correct. Inspect both the final geometry and prediction confidence. The distance report
measures the final model, including any residual deviation after the integrator update.
An explicitly released restraint need not remain exactly at its target.

## Results and reruns

Preview outputs contain structures, confidence plots, the effective input and
`rgi_report.json` with per-seed built counts and measured distances. The download ZIP
includes these files. Output names include model, RGI/vanilla mode, inputs, seeds and
sampling settings, so changing them cannot reuse another run's results under `skip`.
Switch **use_rgi** off and rerun the restraint and prediction cells to remove all RGI
configuration and entity opt-ins. The vanilla process runs the original upstream CLI.
For Boltz-1, rerun installation too: guided and vanilla packages use separate Python 3.12
environments. Its YAML input is included in the ZIP and its native setup/finalize log
provides the restraint inventory.

## Local execution and implementation

The preview integration is pinned to `alphafold3-colabfold==3.1.11`, whose wheels require
Python 3.13. Install its runtime
dependencies as in the notebook, this branch of ColabFold, and RGI-toolkit, then run:

```sh
uv run --no-project python -m colabfold.rgi run_alphafold.py \
  --json_path=input.json --model=boltz2 --norun_data_pipeline \
  --output_dir=results --flash_attention_implementation=xla
```

The input uses the usual AF3 JSON with optional top-level `restraints_config` and
per-entity `conformer_restraints: true`. Omitting the config takes the vanilla path.
Job names must be distinct when configurations differ. The wrapper preserves the
upstream input and output pipeline and uses the existing AF3 adapter. For OpenDDE it
gathers structural-token coordinates into residue order and scatters the correction back.

The installed sampler is checked against the tested source SHA256 before inserting
four in-memory hook points. It retains the upstream schedule, random keys, alignment,
Euler updates and Chai's second denoiser. RGI sees pre-churn sigma and a zero-based step
index. No installed predictor files are changed by these hooks. The minimizer is a JAX
argument, so compatible inputs reuse compilation without capturing the previous job's
restraint values. Unknown sampler versions fail rather than silently omitting RGI.

## Verification

`scripts/rgi_e2e.py` runs real paired predictions, checks finite written CIF coordinates,
measures centroid distances independently with Gemmi, and requires a nonzero inventory.
`--fixture combined` also exercises a trans-fumarate ligand's bond, angle, cis/trans and
plane restraints. Generated environments, weights, notebooks with outputs and predictions
belong under `.cache` or the ignored output paths.

`scripts/rgi_batch_e2e.py` checks vanilla, two distance targets, a custom energy and
vanilla again in one process, with two seeds and two samples per job. It requires the
two vanilla results to agree and each guided job to reach its own target.
`scripts/rgi_boltz1_e2e.py` checks the legacy notebook's YAML with the native Boltz fork.

```sh
uv run --no-project python scripts/rgi_e2e.py --models boltz2
uv run --no-project python scripts/rgi_e2e.py --models opendde --fixture combined
uv run --no-project python scripts/rgi_batch_e2e.py
```

Colab CLI 0.6.0 needs `jupyter-kernel-client<1`; version 1 removed the `KernelClient`
API it imports. A separate uv environment can supply this compatibility pin without
changing an existing CLI installation. For a Colab notebook E2E, set
`AF3_NB_OVERRIDES` to a JSON object in the session, then execute the notebook with
`colab exec -s SESSION -f ColabFold2_preview.ipynb`. Stop that session after collecting
the results. Most model-matrix work can run on a local GPU or a Slurm compute node.
