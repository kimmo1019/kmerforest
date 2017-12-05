Note that data should be libsvm format
festlearn is called this way:
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

festclassify is called this way:

            festclassify [options] data model predictions
            Available options:
                 -t <int>  : number of trees to use (default: 0 = all)


