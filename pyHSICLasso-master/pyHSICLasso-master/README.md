# pyHSICLasso
[![pypi](https://img.shields.io/pypi/v/pyHSICLasso.svg)](https://pypi.python.org/pypi/pyHSICLasso)
[![MIT License](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](LICENSE)
[![Build Status](https://travis-ci.org/riken-aip/pyHSICLasso.svg?branch=master)](https://travis-ci.org/riken-aip/pyHSICLasso)

pyHSICLasso is a package of the Hilbert Schmidt Independence Criterion Lasso (HSIC Lasso), which is a nonlinear feature selection method considering the nonlinear input and output relationship. HSIC Lasso can be regarded as a convex variant of widely used minimum redundancy maximum relevance (mRMR) feature selection algorithm. 

## Advantage of HSIC Lasso

- Can find nonlinearly related features efficiently.
- Can find non-redundant features.
- Can obtain a globally optimal solution.
- Can deal with both regression and classification problems through kernels. 

## Feature Selection
The goal of supervised feature selection is to find a subset of input features that are responsible for predicting output values. By using this, you can supplement the dependence of nonlinear input and output and you can calculate the optimal solution efficiently for high dimensional problem. The effectiveness are demonstrated through feature selection experiments for classification and regression with thousands of features. Finding a subset of features in high-dimensional supervised learning is an important problem with many real- world applications such as gene selection from microarray data, document categorization, and prosthesis control.

## Install
```sh
$ pip install -r requirements.txt
$ python setup.py install
```

or  

```sh
$ pip install pyHSICLasso
```

## Usage
First, pyHSICLasso provides the single entry point as class `HSICLasso()`

This class has the following methods.

- input
- regression
- classification
- dump
- plot_path
- plot_dendrogram
- plot_heatmap
- get_features
- get_features_neighbors
- get_index
- get_index_score
- get_index_neighbors
- get_index_neighbors_score

The input format corresponds to the following formats.

- MATLAB file (.mat)
- .csv
- .tsv
- python's list
- numpy's ndarray

## Input file
When using .mat, .csv, .tsv, we support pandas dataframe. 
The rows of the dataframe are sample number. The output variable should have `class` tag. 
If you wish to use your own tag, you need to specify the output variables by list (`output_list=['tag']`) 
The remaining columns are values of each features. The following is a sample data (csv format). 

```
class,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10
-1,2,0,0,0,-2,0,-2,0,2,0
1,2,2,0,0,-2,0,0,0,2,0
...

```

When using python's list or numpy's ndarray, Let each index be sample number, let values of each features for X[ind] and classification value for Y[ind].

For multi-variate output cases, you can specify the output by using the list (`output_list`). See [Sample code](https://github.com/riken-aip/pyHSICLasso/blob/master/example/sample_multi_variate_output.py) for details.

## To handle large number of samples 
HSIC Lasso scales well with respect to the number of features `d`. However, the vanilla HSIC Lasso requires `O(dn^2)` memory space and may run out the memory if the number of samples `n` is more than 1000. In such case, we can use the block HSIC Lasso which requires only `O(dnBM)` space, where `B << n` is the block parameter and `M` is the permutation parameter to stabilize the final result. This can be done by specifying `B` and `M` parameters in the regression or classification function. Our recommendation would be `B=20` and `M=3`. 

## Example

```py
>>> from pyHSICLasso import HSICLasso
>>> hsic_lasso = HSICLasso()

>>> hsic_lasso.input("data.mat")

>>> hsic_lasso.input("data.csv")

>>> hsic_lasso.input("data.tsv")

>>> hsic_lasso.input([[1, 1, 1], [2, 2, 2]], [0, 1])

>>> hsic_lasso.input(np.array([[1, 1, 1], [2, 2, 2]]), np.array([0, 1]))
```

You can specify the number of subset of feature selections with arguments `regression` and` classification`.

```py
>>> hsic_lasso.regression(5)

>>> hsic_lasso.classification(10)
```

About output method, it is possible to select plots on the graph, details of the analysis result, output of the feature index.

```py
>>> hsic_lasso.plot()
# plot the graph

>>> hsic_lasso.dump()
============================================== HSICLasso : Result ==================================================
| Order | Feature      | Score | Top-5 Related Feature (Relatedness Score)                                          |
| 1     | 1100         | 1.000 | 100          (0.979), 385          (0.104), 1762         (0.098), 762          (0.098), 1385         (0.097)|
| 2     | 100          | 0.537 | 1100         (0.979), 385          (0.100), 1762         (0.095), 762          (0.094), 1385         (0.092)|
| 3     | 200          | 0.336 | 1200         (0.979), 264          (0.094), 1482         (0.094), 1264         (0.093), 482          (0.091)|
| 4     | 1300         | 0.140 | 300          (0.984), 1041         (0.107), 1450         (0.104), 1869         (0.102), 41           (0.101)|
| 5     | 300          | 0.033 | 1300         (0.984), 1041         (0.110), 41           (0.106), 1450         (0.100), 1869         (0.099)|
>>> hsic_lasso.get_index()
[1099, 99, 199, 1299, 299]

>>> hsic_lasso.get_index_score()
array([0.09723658, 0.05218047, 0.03264885, 0.01360242, 0.00319763])

>>> hsic_lasso.get_features()
['1100', '100', '200', '1300', '300']

>>> hsic_lasso.get_index_neighbors(feat_index=0,num_neighbors=5)
[99, 384, 1761, 761, 1384]

>>> hsic_lasso.get_features_neighbors(feat_index=0,num_neighbors=5)
['100', '385', '1762', '762', '1385']

>>> hsic_lasso.get_index_neighbors_score(feat_index=0,num_neighbors=5)
array([0.9789888 , 0.10350618, 0.09757666, 0.09751763, 0.09678892])



```

## References
-  Yamada, M., Tang, J., Lugo-Martinez, J., Hodzic, E., Shrestha, R., Saha, A., Ouyang, H., Yin, D., Mamitsuka, H., Sahinalp, C., Radivojac, P., Menczer, F., & Chang, Y. [Ultra High-Dimensional Nonlinear Feature Selection for Big Biological Data.
](https://ieeexplore.ieee.org/document/8248802/) IEEE Transactions on Knowledge and Data Engineering (TKDE), pp.1352-1365, 2018.
- Yamada, M., Jitkrittum, W., Sigal, L., Xing, E. P. & Sugiyama, M. [High-Dimensional Feature Selection by Feature-Wise Kernelized Lasso.](http://www.ms.k.u-tokyo.ac.jp/2014/HSICLasso.pdf) Neural Computation, vol.26, no.1, pp.185-207, 2014.

## Contributors
### Developers
Name : Makoto Yamada, Héctor Climente-González

E-mail : makoto.yamada@riken.jp

### Distributor
Name : Hirotaka Suetake

E-mail : hirotaka.suetake@riken.jp
