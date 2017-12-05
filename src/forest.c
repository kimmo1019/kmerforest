/*df*/
#include "tree.h"
#include "forest.h"
#include <stdlib.h>
#include <math.h>

void initForest(forest_t* f, int committee, int maxdepth, float param, int trees, float wneg, int oob){
    f->committee = committee;
    f->maxdepth = maxdepth;
    f->factor = param;
    f->ntrees = trees;
    f->ngrown = 0;
    f->wneg = wneg;
    f->oob = oob;
}

void freeForest(forest_t* f){
    int i;
    for(i=0; i<f->ngrown; i++)
        freeTree(f->tree[i]);
    free(f->tree);
}

void tabulateOOBVotes(tree_t* tree, dataset_t* d) {
    // classify the out-of-bag examples and record the votes
    classifyOOBData(tree,tree->root,d);
    int i;
    for(i=0; i < d->nex; i++) {
        if(d->weight[i] == 0) {
            if (tree->pred[i] > 0.5)
                d->oobvotes[i]+=1;
            else
                d->oobvotes[i]-=1;
        }
    }
}

void reportOOBError(dataset_t* d, int iter) {
    float tp,fp,tn,fn;
    int confusion[2][2]={{0,0},{0,0}};
    int i;
    for (i=0; i < d->nex; i++) {
        if(d->oobvotes[i] == 0)
            continue;
        int margin = d->oobvotes[i] > 0 ? 1 : 0;
        confusion[d->target[i]][margin]+=1;
    }
    tp=confusion[1][1];
    fn=confusion[1][0];
    fp=confusion[0][1];
    tn=confusion[0][0];

    float acc = (tp+tn) / (tp+tn+fp+fn);
    float sens = tp / (tp+fn);  // acc on pos examples = recall
    float spec = tn / (tn+fp);  // acc on neg examples
    printf("%5d  %5.2f%%  %5.2f%%  %5.2f%%\n", iter+1, 100*(1-acc), 100*(1-spec), 100*(1-sens));
}

void reportOOBHeader() {
    printf("Error rate (1-acc), on neg examples (1-spec), and on pos examples (1-sens)\n");
    printf("%5s  %6s  %6s  %6s\n","tree","err","negerr","poserr");
}

void growForest(forest_t* f, dataset_t* d,char** OOBindex){
    int i,t,r;
    tree_t tree;
    float sum,c[2],w[2];

    f->nfeat = d->nfeat;
    f->tree = malloc(f->ntrees*sizeof(node_t*));
    tree.valid = malloc(d->nex*sizeof(int));
    tree.used = calloc(d->nfeat,sizeof(int));
    tree.feats = malloc(d->nfeat*sizeof(int));
    /*OOBindex = malloc(f->ntrees*sizeof(char*));
    for(i = 0;i<f->ntrees;i++){
        OOBindex[i] = calloc(d->nex,sizeof(char));
    }*/
    for(i = 0; i<d->nfeat; i++)
        tree.feats[i]=i;
    tree.maxdepth = f->maxdepth;
    tree.committee = f->committee;
    tree.pred = malloc(d->nex*sizeof(float));

    c[0]=c[1]=0;
    for(i=0; i<d->nex; i++){
        c[d->target[i]]+=1;
    }
    w[0]=f->wneg/(f->wneg*c[0]+c[1]);
    w[1]=1.0/(f->wneg*c[0]+c[1]);

    if (f->committee == BOOSTING){
        for(i=0; i<d->nex; i++){
            tree.valid[i]=1;
            d->weight[i]=w[d->target[i]];
        }
    }
    if(f->oob)
        reportOOBHeader();
    if(f->committee == RANDOMFOREST)
        tree.fpn=(int)(f->factor*sqrt(d->nfeat));
    else
        tree.fpn = d->nfeat;
    for(t=0; t<f->ntrees; t++){
       // printf("growing tree %d\n",t);
        if (f->committee == BOOSTING){
            grow(&tree, d);
            classifyTrainingData(&tree, tree.root, d);
            sum=0.0f;
            for(i=0; i<d->nex; i++){
                d->weight[i]*=exp(-(2*d->target[i]-1)*tree.pred[i]);
                sum+=d->weight[i];
            }
            for(i=0; i<d->nex; i++)
                d->weight[i]/=sum;
        }
        else{
            /* Bootstrap sampling */ 
            for(i=0; i<d->nex; i++){
                tree.valid[i]=0;
                d->weight[i]=0;
            }
            for(i=0; i<d->nex; i++){
                r = rand()%d->nex;
                tree.valid[r] = 1;
                d->weight[r] += w[d->target[r]];
            }
            for(i=0; i<d->nex; i++){
                if(d->weight[i]==0)
                    OOBindex[t][i] = 1;
            }
            grow(&tree, d);
            if(f->oob){
                for(i=0; i<d->nex; i++){
                    tree.valid[i] = 1;
                }
                tabulateOOBVotes(&tree, d);
                reportOOBError(d, t);
            }
            memset(d->oobvotes,0,d->nex*sizeof(float));
        }
        f->tree[t] = tree.root;
        f->ngrown += 1;
    }

    free(tree.pred);
    free(tree.valid);
    free(tree.used);
    free(tree.feats);

}


void evaluateFeature(char* input,forest_t* f,dataset_t* d,int maxline,char** OOBindex){
    FILE* fp,*fr;                       /*fp:input file pointer,fr:output file pointer*/
    float* errorRate1;                  /*the OOB error rate of each tree*/
    float** errorRate2;                 /*the error rate of each tree with each dimension randomly changed*/
    float ave_errorRate1;               /*average error rate of OOB data */
    float* ave_errorRate2;              /*average error rata of OOB data with each dimension randomly changed*/
    float confusion1[2][2];             /*confusion matrix of OOB data*/
    float confusion2[2][2];             /*confusion matrix of OOB data with one dimension randomly changed*/
    float TP1,FN1,FP1,TN1;              
    float TP2,FN2,FP2,TN2;
    float* example1;                    /*feature vector of one OOB example*/
    float* example2;                    /*feature vector of one OOB example with one dimension randomly changed*/
    float acc1;
    float acc2;
    float pre;
    float sum;
    float val;
    char** featureMat;                  /*load input file into memory*/
    char** preMat;                      /*prediction of each example with each dimension randomly changed: 0 or 1*/ 
    char* line;
    int i,t,dim;                        /*dim starts from 1*/
    int len;
    int offset;
    int *target;
    int feat;
    int margin;
    int lineCount = -1;
    /*file path test*/
    fp = fopen(input,"r");
    if(fp == NULL){
        printf("Can't open file: %s.\n",input);
        exit(1);
    }
    /*assign memory space*/
    target = malloc(sizeof(int)*d->nex);
    example1 = malloc(d->nfeat*sizeof(float));
    example2 = malloc(d->nfeat*sizeof(float));
    line = malloc(maxline*sizeof(char));
    featureMat = malloc(sizeof(char*)*d->nex);
    errorRate1 = malloc(sizeof(float)*f->ntrees);
    errorRate2 = malloc(sizeof(float*)*d->nfeat);
    for(i=0;i<d->nfeat;i++)
        errorRate2[i]=malloc(f->ntrees*sizeof(float));
    preMat = malloc(d->nex*sizeof(char*));
    for(i=0;i<d->nex;i++)
        preMat[i] = calloc(d->nfeat,sizeof(char));
    ave_errorRate2=malloc(d->nfeat*sizeof(float));
    /*load input file into featureMat*/
    while(fgets(line,maxline,fp)!=NULL){
        lineCount+=1;
        featureMat[lineCount] = malloc(maxline*sizeof(char));
        sscanf(line,"%d%n",target+lineCount,&len);
        *(target+lineCount) = *(target+lineCount)<=0?0:1;
        memcpy(featureMat[lineCount],line,maxline*sizeof(char));
    }
    fclose(fp);
    for(t=0;t<f->ntrees;t++){
        confusion1[0][0]=0;
        confusion1[0][1]=0;
        confusion1[1][0]=0;
        confusion1[1][1]=0;
        for(i=0;i<d->nex;i++){
            if(OOBindex[t][i] != 1)
                continue;
            memset(example1,0,d->nfeat*sizeof(float));
            for(offset = len;sscanf(featureMat[i]+offset,"%d:%f%n",&feat,&val,&len)>=2;offset+=len){
                if(feat<d->nfeat)
                    example1[feat]=val;
            }
            pre = classifyBag(f->tree[t], example1);
            margin = pre > 0.5? 1 : 0;
            confusion1[*(target+i)][margin] += 1;
            for(dim=1;dim<d->nfeat;dim++){
                memcpy(example2, example1, d->nfeat*sizeof(float));
                example2[dim] = rand()%f->nfeat;//how to randomly change its value.
                pre = classifyBag(f->tree[t], example2);
                margin = pre > 0.5 ? 1 : 0;
                preMat[i][dim] = margin>0.5?1:0;
            }

        }
        TP1 = confusion1[1][1];
        FN1 = confusion1[1][0];
        FP1 = confusion1[0][1];
        TN1 = confusion1[0][0];
        acc1 = (TP1+TN1)/(TP1+TN1+FP1+FN1);
        errorRate1[t] = 1-acc1;
        for(dim=1;dim<d->nfeat;dim++){
            confusion2[0][0] = 0;
            confusion2[0][1] = 0;
            confusion2[1][0] = 0;
            confusion2[1][1] = 0;
            for(i=0;i<d->nex;i++){
                if(OOBindex[t][i]!=1)
                    continue;
                confusion2[*(target+i)][preMat[i][dim]]+=1;
            }
            TP2 = confusion2[1][1];
            FN2 = confusion2[1][0];
            FP2 = confusion2[0][1];
            TN2 = confusion2[0][0];
            acc2 = (TP2+TN2)/(TP2+TN2+FP2+FN2);
            if(dim%100==0){
            printf("The %dth tree: %dth dimension of example\n",t,dim);
            printf("%5.0f  %5.0f  %5.0f  %5.0f. Total:%5.0f\n",TP2,FN2,FP2,TN2,TP2+FN2+FP2+TN2);
            }
            errorRate2[dim][t] = 1 - acc2;
        }

    }
    for (dim=1;dim<d->nfeat;dim++){
        sum = 0;
        for(t=0;t<f->ntrees;t++){
            sum += errorRate2[dim][t];
        }
        ave_errorRate2[dim] = sum/f->ntrees;
    }
    sum = 0;
    for(t=0;t<f->ntrees;t++)
        sum+=errorRate1[t];
    ave_errorRate1 = sum/f->ntrees;
    fr = fopen("Feat_importance.txt","w");
    for(dim=1;dim<d->nfeat;dim++){
        //printf("%f  %f\n",ave_errorRate2[dim],ave_errorRate1);
        fprintf(fr,"%d %f\n",dim, ave_errorRate2[dim]-ave_errorRate1);
    }
    fclose(fr);
    /*free memory*/
    free(example1);
    free(example2);
    free(target);
    free(ave_errorRate2);
    for(i=0;i<d->nex;i++)
        free(preMat[i]);
    free(preMat);
    for(i=0;i<d->nex;i++)
        free(featureMat[i]);
    free(featureMat);
    for(i=0;i<f->ntrees;i++)
        free(OOBindex[i]);
    free(OOBindex);
    free(errorRate1);
    for(i=0;i<d->nfeat;i++)
        free(errorRate2[i]);
    free(errorRate2);

}



float classifyForest(forest_t* f, float* example){
    int i;
    float sum = 0;
    if(f->committee == BOOSTING){
        for(i=0; i<f->ngrown; i++){
            sum += classifyBoost(f->tree[i], example);
        }
    }
    else{
        for(i=0; i<f->ngrown; i++){
            sum += classifyBag(f->tree[i], example);
        }
    }
    return sum/f->ngrown;
}

void writeForest(forest_t* f, const char* fname){
    int i;
    char* committeename[8];
    FILE* fp = fopen(fname,"w");
    if(fp == NULL){
        fprintf(stderr,"could not write to output file: %s\n",fname);
        return;
    }
    committeename[BAGGING]="Bagging";
    committeename[BOOSTING]="Boosting";
    committeename[RANDOMFOREST]="RandomForest";

    fprintf(fp, "committee: %d (%s)\n",f->committee, committeename[f->committee]);
    fprintf(fp, "trees: %d\n", f->ngrown);
    fprintf(fp, "features: %d\n", f->nfeat);
    fprintf(fp, "maxdepth: %d\n", f->maxdepth);
    fprintf(fp, "fpnfactor: %g\n", f->factor);
    for(i=0; i<f->ngrown; i++){
        writeTree(fp,f->tree[i]);
    }
    fclose(fp);
}

void readForest(forest_t* f, const char* fname){
    int i;
    FILE* fp = fopen(fname,"r");
    if(fp == NULL){
        fprintf(stderr,"could not read input file: %s\n",fname);
        exit(1);
    }
    fscanf(fp, "%*s%d%*s",&f->committee);
    fscanf(fp, "%*s%d", &f->ngrown);
    fscanf(fp, "%*s%d", &f->nfeat);
    fscanf(fp, "%*s%d", &f->maxdepth);
    if(fscanf(fp, "%*s%g", &f->factor)==EOF) {
        fprintf(stderr,"corrupt input file: %s\n",fname);
        exit(1);
    }
    f->tree = malloc(sizeof(node_t*)*f->ngrown);
    for(i=0; i<f->ngrown; i++){
        readTree(fp,&(f->tree[i]));
    }
    if(fscanf(fp, "%*s")!=EOF){
        fprintf(stderr,"garbage at the end of input file: %s\n",fname);
    }
    fclose(fp);
}

