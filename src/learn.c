
#include "dataset.h"
#include "tree.h"
#include "forest.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <getopt.h>


int main(int argc, char* argv[]){
    dataset_t d;
    forest_t f;
    int option,i;
    int reportoob=0;
    int trees=100;
    int maxdepth=1000;
    int committee=3;
    int maxline;
    float param=1.0f;
    float w=1.0;
    char* input=0;
    char* model=0;
    char** OOBindex;   /*OOB example index of each tree*/
    time_t tim;
    
    const char* help="Usage: %s [options] data model\nAvailable options:\n\
    -c <int>  : committee type:\n\
                1 bagging\n\
                2 boosting\n\
                3 random forest(default)\n\
    -d <int>  : maximum depth of the trees (default: 1000)\n\
    -e        : report out of bag estimates (default: no)\n\
    -n <float>: relative weight for the negative class (default: 1)\n\
    -p <float>: parameter for random forests: (default: 1)\n\
                (ratio of features considered over sqrt(features))\n\
    -t <int>  : number of trees (default: 100)\n";
    

    while((option=getopt(argc,argv,"c:d:en:p:t:"))!=EOF){
        switch(option){
            case 'c': committee=atoi(optarg); break;
            case 'd': maxdepth=atoi(optarg); break;
            case 'e': reportoob=1; break;
            case 'n': w=atof(optarg); break;
            case 'p': param=atof(optarg); break;
            case 't': trees=atoi(optarg); break;
            case '?': fprintf(stderr,help,argv[0]); exit(1); break;
        }
    }
    if(committee!=BAGGING && committee!=BOOSTING && committee!=RANDOMFOREST){
        fprintf(stderr,"Unknown committee type\n");
        exit(1);
    }
    if(maxdepth<=0){
        fprintf(stderr,"Invalid tree depth\n");
        exit(1);
    }
    if(w<0){
        fprintf(stderr,"Invalid weight for negative class\n");
        exit(1);
    }
    if(param<=0){
        fprintf(stderr,"Invalid parameter value\n");
        exit(1);
    }
    if(trees<=0){
        fprintf(stderr,"Invalid number of trees\n");
        exit(1);
    }
    if(argc - optind == 2){
        input = argv[optind];
        model = argv[optind+1];
    }
    else{
        fprintf(stderr,help,argv[0]); 
        exit(1);
    }

    tim = time(0);
    srand(tim);
    loadData(input,&d,&maxline);
    initForest(&f,committee,maxdepth,param,trees,w,reportoob);
    OOBindex = (char**)malloc(sizeof(char*)*f.ntrees);
    for (i=0;i<f.ntrees;i++){
        OOBindex[i] = calloc(d.nex, sizeof(char));
    }
   
    growForest(&f, &d,OOBindex);
    writeForest(&f, model);
    evaluateFeature(input,&f,&d,maxline,OOBindex);
    freeForest(&f);
    freeData(&d);
    return 0;
}
