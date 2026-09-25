# Phase 1 report: dataset pool

Profiled 135 datasets (0 failed).

## Pool by category and source

| category   |   extension |   tabrepo |   All |
|:-----------|------------:|----------:|------:|
| binarized  |           6 |        19 |    25 |
| primary    |          26 |        55 |    81 |
| synthetic  |           0 |        29 |    29 |
| All        |          32 |       103 |   135 |

## Possible duplicate datasets (column-fingerprint Jaccard >= 0.5)

|   did_a | name_a    |   did_b | name_b   |   jaccard |   shared_columns |
|--------:|:----------|--------:|:---------|----------:|-----------------:|
|     735 | cpu_small |     761 | cpu_act  |     0.571 |               12 |

## Spread of quality dimensions: primary datasets, TabRepo only

| dimension                           |   median |      p10 |        p90 |        max |   share_nonzero |
|:------------------------------------|---------:|---------:|-----------:|-----------:|----------------:|
| Missing cells (fraction)            |  0       |   0      |   0.04     |   0.698    |           0.182 |
| Rows with any missing value         |  0       |   0      |   0.797    |   1        |           0.182 |
| Duplicate rows (fraction)           |  0.00048 |   0      |   0.451    |   1        |           0.509 |
| Imbalance ratio (majority/minority) |  2.5     |   1.01   |  27        | 242        |         nan     |
| Rows                                |  5.4e+03 | 713      |   9.32e+04 |   1e+06    |         nan     |
| Features                            | 21       |   6      | 247        |   1.78e+03 |         nan     |
| Categorical features (fraction)     |  0       |   0      |   0.66     |   1        |           0.418 |
| Feature pairs with |r| > 0.8        |  0.00549 |   0      |   0.217    |   0.736    |           0.654 |
| Constant features                   |  0       |   0      |   0.6      |  25        |           0.109 |
| Categoricals with > 30 levels       |  0       |   0      |   1.6      |  39        |           0.182 |
| Mean |skewness|                     |  3.67    |   0.751  |  15.8      |  73.7      |           1     |
| Outlier cells (1.5 IQR rule)        |  0.0514  |   0.0153 |   0.102    |   0.242    |           0.962 |

## Spread of quality dimensions: primary datasets, TabRepo + extension

| dimension                           |    median |      p10 |       p90 |        max |   share_nonzero |
|:------------------------------------|----------:|---------:|----------:|-----------:|----------------:|
| Missing cells (fraction)            |  0        |   0      |   0.182   |   0.909    |           0.444 |
| Rows with any missing value         |  0        |   0      |   0.991   |   1        |           0.444 |
| Duplicate rows (fraction)           |  0        |   0      |   0.346   |   1        |           0.444 |
| Imbalance ratio (majority/minority) |  3.03     |   1.02   |  17.5     | 242        |         nan     |
| Rows                                |  5.83e+03 | 690      |   1.5e+05 |   1e+06    |         nan     |
| Features                            | 26        |   6      | 298       |   1.78e+03 |         nan     |
| Categorical features (fraction)     |  0.1      |   0      |   0.857   |   1        |           0.543 |
| Feature pairs with |r| > 0.8        |  0.00996  |   0      |   0.192   |   0.736    |           0.64  |
| Constant features                   |  0        |   0      |   2       | 167        |           0.198 |
| Categoricals with > 30 levels       |  0        |   0      |   5       |  39        |           0.272 |
| Mean |skewness|                     |  3.67     |   0.765  |  17.6     |  73.7      |           1     |
| Outlier cells (1.5 IQR rule)        |  0.0486   |   0.0157 |   0.0988  |   0.242    |           0.974 |
