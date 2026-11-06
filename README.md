# Minimum Sufficient Transparency: model and behaviour

Code and data for the article

> N. Kumar, Y. Mualla, V. Hilaire, S. Galland, M. Dridi.
> **Enough, but No More: Minimum Sufficient Transparency for Human Oversight of Multi-Agent Systems.**
> Manuscript submitted to *Information Sciences* (2026).

Minimum Sufficient Transparency (MST) is the selection capability of the assistant agent in a
human-supervised multi-agent system. For each stakeholder role, it selects the subset of admissible
information items that reaches a sufficiency threshold $U(S) \ge \tau$ while minimizing a weighted
combination of relative privacy exposure, interpretation cost, and size. When no admissible subset is
sufficient, it reports `INFEASIBLE` with the shortfall and the release of prohibited items that would
restore sufficiency. A $\Gamma$-robust counterpart keeps the disclosure sufficient when up to $\Gamma$
item utilities are overestimated.

This repository covers the model paper only: the model, its robust counterpart, the controlled
experiments on its behaviour, the case study on the LBNL Building 59 data set, and the SARL
execution. The algorithmic study of heuristics and metaheuristics for large pools is a separate work.

## Structure

| Folder | Content |
|---|---|
| `mst/` | Reference Python implementation of the set functions and exact enumeration (standard library only). |
| `optimization/` | Exact mixed-integer model, nominal and $\Gamma$-robust, solved with SCIP (`scip_model.py`); reference scales (`scales.py`); simple constructive rules used as baselines (`heuristics.py`); scripts of the estimation-error, policy-strictness, and reference-scale experiments. |
| `comparison/` | Synthetic pool generator (`benchmark_pools.py`), MST, its ablations, and the selection rules of the closest approaches (`methods.py`), and the comparison experiment. |
| `case_study_lbnl/` | Preprocessing of the LBNL Building 59 data and the case study. |
| `sarl/` | SARL and Java implementation: role agents request disclosures from the MST capacity (see `sarl/README.md`). |
| `paper_scripts/` | Scripts that produce every figure and table of the article and its Supplementary Material. |
| `data/` | Results: `benchmark/` (controlled experiments), `comparison/`, `case_study_lbnl/` (derived hourly data and results), `sarl/`. |
| `figures/` | Figures of the article, and the draw.io source of Figure 1. |
| `tables/` | LaTeX rows of the tables of the article and the Supplementary Material. |

## Reproducing the results

```bash
pip install -r requirements.txt            # numpy, pandas, matplotlib, pyscipopt
python comparison/run_comparison.py        # Section 6.4, data/comparison/comparison_rows.csv.gz
python comparison/sensitivity.py           # Table S3
python optimization/run_robustness.py      # Section 6.2, data/benchmark/robustness.csv
python optimization/run_admissibility.py   # Section 6.3, data/benchmark/admissibility.csv.gz
python optimization/run_normalization.py   # Section 6.5, data/benchmark/normalization.csv
python case_study_lbnl/prepare_data.py "<path to Bldg59_clean data>"   # optional, output is included
python case_study_lbnl/run_case_study.py   # Section 7
python paper_scripts/make_figures.py
python paper_scripts/make_tables.py
```

The synthetic pools are generated with fixed seeds. The case study uses the LBNL Building 59 data set
(Luo et al., *Scientific Data* 9, 156, 2022), available on Dryad at
https://doi.org/10.7941/D1N33Q under a CC BY 4.0 licence; the derived hourly data used by the case
study are included in `data/case_study_lbnl/`.

## Licence

Code under the licence in `LICENSE`. The derived building data follow the CC BY 4.0 licence of the
original data set.
