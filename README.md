# Kmerforest
Kmerforest is a sequence-based method to predict the impact of regulatory variants using random forest.

# Usage
This software has been tested in a Linux/MacOs system.
Note that input data should be in [libsvm format](https://www.csie.ntu.edu.tw/~cjlin/libsvm/).

Training:

            festlearn [options] data model
            Available options:
                -c <int>  : committee type:
                            1 bagging
                            2 boosting 
                            3 random forest(default)
                -d <int>  : maximum depth of the trees (default: 1000)
                -e        : report out of bag estimates (default: no)
                -n <float>: relative weight for the negative class (default: 1)
                -p <float>: parameter for random forests: (default: 1)
                            (ratio of features considered over sqrt(features))
                -t <int>  : number of trees (default: 100)

Test:

            festclassify [options] data model predictions
            Available options:
                 -t <int>  : number of trees to use (default: 0 = all)

# Citation
**Liu Q**, Gan M, Jiang R. A sequence-based method to predict the impact of regulatory variants using random forest[J]. *BMC systems biology*, 2017, 11(2): 7.

# License
This project is licensed under the MIT License - see the LICENSE.md file for details
