Subject: Take-home submission — GRN-Prior Expression Embedding Benchmark (Bartek Skorulski)

Hi Caelan,

Here is my submission for the GRN-Prior Expression Embedding Benchmark:

  https://github.com/sbartek/grn-prior-benchmark

Everything needed to review and run it is in the README; the 2-page write-up is in
`memo/memo.md`, and a rendered docs site is at https://sbartek.github.io/grn-prior-benchmark/.

Short version of what I found: on the trustworthy readout (cell type, donor-grouped CV), the
full DoRothEA prior underperforms both PCA and an unconstrained autoencoder. Restricting to
high-confidence edges closes most of the gap, but a density ablation with degree-preserving
rewired controls shows that improvement is mostly a sparsity/regularization effect rather than
regulatory biology; the specific topology adds only a small, consistent margin over its rewired
control at full data and under noise, and never beats the baseline. Disease is not decodable
from held-out donors on this between-subjects dataset, so it can't adjudicate the question. I
kept the comparison capacity-matched and tried to defend any apparent benefit against the
obvious "it's just sparsity" alternative — details and limitations are in the memo.

Reproduce:
  uv venv --python 3.11 && uv pip install -r requirements.txt
  uv run python scripts/00_fetch_data.py      # caches the CELLxGENE subset
  uv run python scripts/01_build_pseudobulk.py
  uv run python scripts/02_build_graph.py
  uv run python scripts/03_run_experiments.py  # + 05_density_ablation.py for the ablation

There's also an interactive EDA notebook (`notebooks/01_eda_suitability.ipynb`, with a
self-contained HTML export) covering dataset suitability and the donor confound.

Happy to walk through any of the choices and results.

Best,
Bartek
