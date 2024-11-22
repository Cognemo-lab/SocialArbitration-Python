# tapas_fitModel


## Variables


### Dictionary `r`

| Key             | Type  | Comment   |
|-----------------|:-----:|:---------:|
| `y`             | n x 1 | responses |
| `u`             | n x 2 | inputs    |
| `c_opt`         |       |           |
| `c_prc`         |       |           |


# dataPrep


## Inputs

| Variables       | Type  | Comment   |
|-----------------|:-----:|:---------:|
| `responses`     | n x 1 | responses |
| `inputs`        | n x 2 | inputs    |


## Outputs

| Variables       | Type       | Comment   |
|-----------------|:----------:|:---------:|
| `r`             | dictionary | data      |


# optim


## Inputs

| Variables       | Type       | Comment                                   |
|-----------------|:----------:|:-----------------------------------------:|
| `r`             | dictionary | data                                      |
| `prc_fun`       | function   | default: `MS9_dmpad_hgf_ar1_lvl3`         |
| `obs_fun`       | function   | default: `MS9_dmpad_constant_voltemp_exp` |
| `opt_algo`      | function   | default: `tapas_quasinewton_optim`        |


## Outputs

| Variables       | Type       | Comment   |
|-----------------|:----------:|:---------:|
| `r`             | dictionary | data      |


# negLogJoint


## Inputs

| Variables       | Type       | Comment                                   |
|-----------------|:----------:|:-----------------------------------------:|
| `r`             | dictionary | data                                      |
| `prc_fun`       | function   | default: `MS9_dmpad_hgf_ar1_lvl3`         |
| `obs_fun`       | function   | default: `MS9_dmpad_constant_voltemp_exp` |
| `ptrans_prc`    | row vec    | `priormus`                                |
| `ptrans_obs`    | row vec    | `priormus`                                |


## Outputs

| Variables       | Type       | Comment   |
|-----------------|:----------:|:---------:|
| `negLogJoint`   | float      |           |
| `negLogLl`      | float      |           |
| `rval`          | float      | 0.        |
| `err`           | array      | empty     |


# optimrun


## Inputs

| Variables       | Type       | Comment                                   |
|-----------------|:----------:|:-----------------------------------------:|
| `nlj`           | function   |                                           |
| `init`          | row vec    |                                           |
| `opt_idx`       | array      | non zero and non nan indices              |
| `opt_algo`      | function   | default: `tapas_quasinewton_optim`        |
| `c_opt`         | opt object | default: `tapas_quasinewton_optim_config` |


## Outputs

| Variables       | Type       | Comment   |
|-----------------|:----------:|:---------:|
| `optres`        | dictionary |           |


## Optimization runs with random initialization

If `nRandInit` is set `> 0` in the optimization algorithm class, further
optimization runs are executed.  The seed of the random numbers can be set
using the `randSeed` attribute.


# restrictfun


## Inputs

| Variables       | Type       | Comment                                   |
|-----------------|:----------:|:-----------------------------------------:|
| `f`             | function   |                                           |
| `arg`           | array      |                                           |
| `free_idx`      | array      | non zero and non nan indices              |
| `free_arg`      | array      | corresponding values                      |


## Outputs

| Variables       | Type       | Comment   |
|-----------------|:----------:|:---------:|
| `val`           | float      |           |


# tapas_quasinewton_optim


## Variables

| Variable        | Scalar | Column Vec | Row Vec | Matrix |
|-----------------|:------:|:----------:|:-------:|:------:|
| `n`             | x      |            |         |        |
| `tolGrad`       | x      |            |         |        |
| `tolArg`        | x      |            |         |        |
| `maxStep`       | x      |            |         |        |
| `maxIter`       | x      |            |         |        |
| `maxRegu`       | x      |            |         |        |
| `maxRst`        | x      |            |         |        |
| `init`          |        | n x 1      |         |        |
| `x`             |        | n x 1      |         |        |
| `val`           | x      |            |         |        |
| `grad`          |        |            | 1 x n   |        |
| `T`             |        |            |         | n x n  |
| `descvec`       |        | n x 1      |         |        |
| `slope`         | x      |            |         |        |
| `newx`          | init.  | n x 1      |         |        |
| `newval`        | x      |            |         |        |
| `dval`          | x      |            |         |        |
| `resetcount`    | x      |            |         |        |
| `stepSize`      | x      |            |         |        |
| `regucount`     | x      |            |         |        |
| `t`             | x      |            |         |        |
| `dx`            |        | n x 1      |         |        |
| `oldgrad`       |        |            | 1 x n   |        |
| `dgrad`         |        |            | 1 x n   |        |
| `dgdx`          | x      |            |         |        |
| `dgT`           |        |            | 1 x n   |        |
| `dgTdg`         | x      |            |         |        |
| `u`             |        | n x 1      |         |        |


### Dictionary `optim`
| Key             | Type   | Comment   |
|-----------------|:------:|:---------:|
| `valMin`        | scalar |           |
| `argMin`        | n x 1  |           |
| `T`             | n x n  |           |
