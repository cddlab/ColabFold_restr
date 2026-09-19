# Configure RGI in ColabFold

Open [ColabFold2 preview](https://colab.research.google.com/github/th2ch-g/ColabFold_restr/blob/rgi-integration/ColabFold2_preview.ipynb).

1. In **1. Install and choose RGI**, enable **use_rgi** and run the cell.
2. Fill in and run **2. Enter molecules**.
3. Run **3. Configure RGI** with its left-hand play button. The editable form opens **below that cell**.
4. Click **Add distance**, **Add conformer**, **Add angle**, **Add custom** or **Add RMSD**, then fill in the new entry. Repeat for multiple restraints.
5. Click **Check RGI settings**, then run **4. Predict structure**.

Leave **use_rgi** off for vanilla. RGI needs an explicit configuration before prediction.
For field-by-field examples, uploads and all native selections, see
[RGI-toolkit's notebook guide](https://github.com/cddlab/rgi_toolkit/blob/main/docs/colabfold.md).
