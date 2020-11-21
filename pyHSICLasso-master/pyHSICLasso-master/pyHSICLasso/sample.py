#!/usr/bin.env python
# coding: utf-8

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import sys
from future import standard_library
#print(sys.path)
from .api import HSICLasso
#import api
import numpy as np

standard_library.install_aliases()


def main():
    hsic_lasso = HSICLasso()
    hsic_lasso.input("./pyHSICLasso/user_data.csv", output_list=['c1', 'c2', 'c3', 'c4', 'c5,', 'c6', 'c7', 'c8', 'c9', 'c10'])
        # ,'c11', 'c12', 'c13', 'c14', 'c15,', 'c16', 'c17', 'c18', 'c19', 'c20','c21', 'c22', 'c23', 'c24', 'c25,', 'c26', 'c27', 'c28', 'c29', 'c30'])
    hsic_lasso.regression(100,B=30)
    hsic_lasso.dump()
    select_index = hsic_lasso.get_index()
    print(select_index)
    print(hsic_lasso.get_index_score())
    #hsic_lasso.plot_path()
    print(hsic_lasso.get_features())
    X_select = hsic_lasso.X_in[select_index, :]
    print(type(X_select))
    np.savetxt('X_select.txt', X_select, fmt=str('%d'), encoding='utf-8')


if __name__ == "__main__":
    main()
