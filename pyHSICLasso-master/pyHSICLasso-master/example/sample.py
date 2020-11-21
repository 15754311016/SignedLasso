#!/usr/bin.env python
# coding: utf-8

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from future import standard_library
#import sys
#sys.path.append("./pyHSICLasso-master/pyHSICLasso")
from pyHSICLasso import HSICLasso
#import api
import numpy as np

standard_library.install_aliases()


def main():
    hsic_lasso = HSICLasso()
    #out_list = ['c'+str(i) for i in range(1,51)]
    #print (out_list)
    hsic_lasso.input("./user_data_new.csv", output_list=['c1','c2','c3','c4','c5,','c6','c7','c8','c9','c10'])
        # ,'c11', 'c12', 'c13', 'c14', 'c15,', 'c16', 'c17', 'c18', 'c19', 'c20','c21', 'c22', 'c23', 'c24', 'c25,', 'c26', 'c27', 'c28', 'c29', 'c30'])
    hsic_lasso.regression(100,B=50)
    hsic_lasso.dump()
    select_index = hsic_lasso.get_index()
    print(select_index)
    print(hsic_lasso.get_index_score())
    #hsic_lasso.plot_path()
    print(hsic_lasso.get_features())
    X_select = hsic_lasso.X_in[select_index, :]
    np.savetxt('X_select.txt', X_select,fmt=str('%.5f'),encoding='utf-8')


if __name__ == "__main__":
    main()
