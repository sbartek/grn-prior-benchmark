# References

Sources cited in the memo's *Related Work* section, with one-line relevance.

**Simple baselines beat complex/biology-wired models**
- Kedzierska et al., *Deeper evaluation of a single-cell foundation model* (scBERT), Nat Mach Intell 2024 — logistic regression matches scBERT on cell typing. https://www.nature.com/articles/s42256-024-00949-w
- Ahlmann-Eltze et al., *Deep-learning perturbation prediction vs linear baselines*, Nat Methods 2025 — "one PCA still rules them all." https://www.nature.com/articles/s41592-025-02772-6

**Biology-wired networks: interpretability, not accuracy; need shuffled controls**
- Lotfollahi et al., *expiMap* — biologically-informed (gene-program masked) decoder, Nat Cell Biol 2023. https://www.nature.com/articles/s41556-022-01072-x
- Elmarakeby et al., *P-NET* — Reactome-wired sparse net, Nature 2021. https://www.nature.com/articles/s41586-021-03922-4
- Kong/Yu et al. (reliability of biology-inspired DNNs) — randomly-wired nets often match knowledge-primed ones ⇒ shuffled/rewired control is required, PMC 2023. https://pmc.ncbi.nlm.nih.gov/articles/PMC10564878/
- Biologically-informed NN performance (simulation + empirical), PMC 2025 — BINNs help mainly under small sample / weak signal. https://pmc.ncbi.nlm.nih.gov/articles/PMC12642320/

**TF-activity as the canonical use of a regulatory prior**
- Aibar et al., *SCENIC* (regulon AUCell activity), Nat Methods 2017. https://www.nature.com/articles/nmeth.4463
- Müller-Dott et al., *CollecTRI*, Nucleic Acids Research 2023 — recommended over DoRothEA. https://academic.oup.com/nar/article/51/20/10934/7318114
- decoupler ULM / DoRothEA docs. https://decoupler.readthedocs.io/

**Evaluation of representation quality + the donor confound**
- Luecken et al., *scIB — benchmarking atlas integration*, Nat Methods 2021 — bio-conservation vs batch-removal metrics; used defaults, not per-dataset HPO. https://www.nature.com/articles/s41592-021-01336-8
- Squair et al., *Confronting false discoveries in single-cell DE*, Nat Commun 2021 — pseudoreplication/donor confound ⇒ aggregate to donor pseudobulk, split by donor. https://www.nature.com/articles/s41467-021-25960-2

**Virtual cell framing (no GRN prior)**
- Arc Institute, *STATE* virtual-cell model — data-driven, uses no hard-coded GRN prior. https://arcinstitute.org/news/virtual-cell-model-state
