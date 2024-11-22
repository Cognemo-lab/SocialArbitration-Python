# MS9_dmpad_hgf_ar1_lvl3


## Variables

| Variable        | Scalar | Column Vec | Row Vec | Matrix  |
|-----------------|:------:|:----------:|:-------:|:-------:|
| `p`             |        |            |   x     |         |
| `u`             |        | n x 1      |         |         |
| `t`             |        | n x 1      |         |         |
| `muhat`         |        |            |         | n x l   |
| `pihat`         |        |            |         | n x l   |
| `w`             |        |            |         | n x l-1 |
| `da`            |        |            |         | n x l   |
| `mu_0`          |        |            |   x     |         |
| `sa_0`          |        |            |   x     |         |
| `phi`           |        |            |   x     |         |
| `m`             |        |            |   x     |         |
| `ka`            |        |            |   x     |         |
| `om`            |        |            |   x     |         |
| `th`            | x      |            |         |         |
| `mu`            |        |            |         | n x l   |
| `pi`            |        |            |         | n x l   |


# MS9_dmpad_hgf_ar1_lvl3_transp

## Variables

| Variable        | Type       |
|-----------------|:----------:|
| `pvec`          | row vec    |
| `pstruct`       | dictionary |
| `l`             | int        |


### `pstruct`

| Key             | Scalar | Row Vec |
|-----------------|:------:|:-------:|
| `mu_0`          |        | 1 x l   |
| `sa_0`          |        | 1 x l   |
| `phi`           |        | 1 x l   |
| `m`             |        | 1 x l   |
| `ka`            |        | 1 x l-1 |
| `om`            |        | 1 x l-1 |
| `th`            | x      |         |
