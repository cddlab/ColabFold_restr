# RGI in the standard Colab form

Open [ColabFold2 preview](https://colab.research.google.com/github/th2ch-g/ColabFold_restr/blob/rgi-integration/ColabFold2_preview.ipynb#scrollTo=rgi-restraints).
The original headings and workflow are retained. **RGI (optional)** is a standard
Colab form after **Input sequences**, visible before execution.

1. Choose an RGI-supported model such as **boltz2** in **Install dependencies**.
2. Fill in **Input sequences**.
3. In **RGI (optional)**, turn on **use_rgi** and enter the selections and targets.
4. Use **Runtime → Run all**.

For a 25 Å distance, set `distance_atom_selection1` to `chain A and resid 1 to 10`,
`distance_atom_selection2` to `chain A and resid 40 to 50`, and `target_distance` to
`25`. Use chains and residues from your own input. Leave unused restraint fields empty.

The same form provides conformer, angle, custom and RMSD. Lists in the fields create
multiple restraints; `restraints_config` accepts additional native toolkit entries
or `{"config_path": "restraints.yaml"}`. See the
[full guide and examples](https://github.com/cddlab/rgi_toolkit/blob/main/docs/colabfold.md).

Leave **use_rgi** off for vanilla. AF2, OpenBind0 and IntelliFold2 are vanilla only.
After edits, rerun the RGI cell and prediction, or use **Run all**. In Boltz-1 the form
precedes installation; rerun installation too when changing **use_rgi**.
