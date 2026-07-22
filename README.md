# Analysis package for Japanese eel swim-tunnel manuscript

This repository uses two processed CSV files as the direct analysis inputs:

- `data/processed/individual_endpoints.csv`: individual/trial-level endpoint data
- `data/processed/speed_step_data.csv`: speed-step-level MO2 and COT data

The analysis scripts validate these CSV inputs and generate statistical summaries,
figure-ready summaries, and optional figures.

## Directory structure

```text
swim-tunnel_respirometry_experiments_in_Japanese_eels/
  README.md
  requirements.txt
  run_all.py
  scripts/
    common.py
    01_validate_inputs.py
    02_reanalysis_individual_endpoints.py
    03_speed_step_summaries.py
    04_make_figures.py
  data/
    processed/
      individual_endpoints.csv
      speed_step_data.csv
  outputs/
    tables/
    figures/
```

## Setup

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

Copy the two source Excel files into `data/raw/`:

```text
data/raw/individual_endpoints.xlsx
data/raw/speed_step_data.xlsx
```

## Run all analyses

From the package root:

```bash
python run_all.py
```

This will produce:

```text
outputs/tables/Table_S4_statistical_summary.csv
outputs/tables/Table_S4_statistical_summary.xlsx
outputs/tables/Table_S4_body_size_sensitivity.csv
outputs/tables/Table_S4_body_size_sensitivity.xlsx
outputs/tables/speed_n_summary.csv
outputs/tables/mo2_speed_summary_by_condition.csv
outputs/tables/cot_speed_summary_by_condition.csv
outputs/figures/fig_mo2_speed_by_condition.png
outputs/figures/fig_cot_speed_by_condition.png
```

## Main statistical framework

The analysis is deliberately not formulated as a full factorial model because the experiment was not fully factorial. Instead, the scripts implement three prespecified contrasts:

1. **FW25 vs FW18 in yellow eels**  
   Paired comparison of sequential trials in the same individuals. The estimate is FW25 minus FW18. This comparison should be described as a temperature-associated difference under fixed trial order.

2. **Y2 FW18 vs Y2 SW18**  
   Independent exploratory freshwater-seawater comparison at 18 °C. The estimate is SW18 minus FW18. This should be described as a salinity-associated comparison, not as a definitive salinity effect.

3. **SW18 Y2 vs SW18 S1**  
   Independent stage-associated comparison under SW18. The estimate is S1 minus Y2. This should be interpreted cautiously because silvering stage and body size are partly confounded.

For each response variable, the scripts report:

- group sample sizes
- means, SDs, and medians
- effect estimate
- unadjusted 95% permutation confidence interval obtained by inversion of the corresponding exact permutation test
- exact paired sign-flip p-value for paired contrasts or exact two-sample label-permutation p-value for independent-group contrasts
- Benjamini-Hochberg FDR-adjusted p-value within each comparison and analysis set
- exclusion counts and notes

For independent-group contrasts, confidence intervals are obtained under an additive location-shift framework. Exact confidence sets that are unbounded because of very small sample size are reported as unbounded. Confidence intervals are unadjusted; FDR adjustment is applied only to p-values.

## Inclusion rules

### Ucrit variables

Used when `ucrit_status` is one of:

```text
valid
boundary_ambiguous
protocol_check
```

Excluded when status is missing, `not_estimable`, or `excluded`.

### Uopt, minCOT, and MO2 at Uopt

Main analysis uses:

```text
valid
boundary_estimate
```

Sensitivity analysis uses only:

```text
valid
```

Rows labelled `single_active_point`, `not_estimable`, or `excluded` are excluded.

### Low-flow MO2

Only rows with:

```text
lowflow_mo2_status = valid
```

are used.

## Body-size sensitivity analysis

For the SW18 Y2 vs S1 comparison, the scripts fit exploratory OLS models:

```text
response ~ stage_S1 + TL
response ~ stage_S1 + BW
```

TL and BW are not entered simultaneously. These models are exploratory and should be used to assess sensitivity of stage-associated differences to body-size adjustment, not to make strong causal claims.

## Reproducibility note

For the sample sizes in the present dataset, all sign patterns or group-label allocations are enumerated exactly. Therefore, the reported main permutation results do not depend on a random seed. The seed and n-permutation arguments are retained only for Monte Carlo fallback when the exact permutation space is too large to enumerate.


## LICENSE

Code is released under the MIT License; data are released under CC BY 4.0
