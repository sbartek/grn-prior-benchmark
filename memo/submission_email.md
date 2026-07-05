Subject: Take-home submission — GRN-Prior Expression Embedding Benchmark (Bartek Skorulski)

Hi Caelan,

Here is my submission for the GRN-Prior Expression Embedding Benchmark:

  Repo:  https://github.com/sbartek/grn-prior-benchmark
  Memo:  memo/memo.pdf  (3-page write-up; source in memo/memo.md)
  Site:  https://sbartek.github.io/grn-prior-benchmark/  (detailed results + a short wiki on the
         models and evaluation techniques)

Short version of what I found: how you *use* the graph matters more than whether you use it.
Baking DoRothEA into a learned encoder (hard or soft mask) hurts, and a degree-preserving rewired
graph does just as well, so that effect is regularization rather than biology. Using the same graph
the classical way — as a fixed decoupler ULM TF-activity transform — carries genuine signal (it
beats both a rewired-network and a random-projection control) and wins specifically in the low-data
regime; CollecTRI does this a bit better than DoRothEA. Interestingly the winner depends on the
metric: a plain PCA baseline tops the supervised probe, but the GRN-informed representations win the
stricter unsupervised clustering test. On the primary RA data disease is confounded with donor, so I
treat cell type as the trustworthy readout; on the COVID backup (3 disease states) disease is
decodable but still partly donor identity. Results replicate on that second dataset. Full reasoning,
controls, and limitations are in the memo.

How to review / run: the README has the setup and one-command-per-step pipeline (uv + Python 3.11).
Three notebooks tell the story end to end: 01 (EDA & dataset suitability), 02 (a tiny toy example
building the intuition), 03 (the full evaluation). The detailed numbers and figures are on the site
linked above.

I tried to keep the comparison fair and honest — scrambled-graph controls throughout, held-out
donors, and reporting the cases where the prior loses as plainly as where it wins. Happy to walk
through any of the choices and results.

Best,
Bartek
