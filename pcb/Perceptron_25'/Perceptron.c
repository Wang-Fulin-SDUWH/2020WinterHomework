#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
//fgets(char *buf, int n, FILE *fp)函数：碰到\n会终止，\n之前读取的n-1个字符的内容放到缓冲区buf里
//strdup()在内部调用了malloc()为变量分配内存，不需要使用返回的字符串时，需要用free()释放相应的内存空间，否则会造成内存泄漏。
//strdup()返回一个指针,指向为复制字符串分配的空间;如果分配空间失败,则返回NULL值。
//例子：指针 ptr 的类型是 type1,它指向的类型是 type2，它被初始化为指向整型变量 a。
//若指针 ptr 被加了 1，编译器是这样处理的：它把指针 ptr 的值加上了 sizeof(type2).
//因此动态分配内存的时候要给ptr分配n*sizeof(type2)，n为数组元素个数



/*公共部分函数声明*/
void get_two_dimension(char* line, double** data, char *filename);
void print_two_dimension(double** data, int row, int col);
double accuracy_metric(double* actual,double* predicted,int len_actual);
double *evaluate_algorithm(double **data, int lendata);
double ***k_cross_validation(double **data, int lendata);
int get_row(char *filename);
int get_col(char *filename);

/*全局变量*/
static int n_folds;//为了避免麻烦，多个参数都要使用n_folds这一参数，直接设为全局变量。
static int row, col;//CSV文件的行、列数
static int fold_size;//fold_size是数据集被分成k块后每一块的长度
static double l_rate=0.1;//定义学习率
static int n_epoch = 100;//循环次数


/*Perceptron函数声明*/
double predict(double *row,double *weights);
double *train_weights(double **train, double l_rate, int n_epoch);
double *perceptron(double **train, double **test, double l_rate, int n_epoch);

int main()
{
    printf("请输入交叉检验分成的块数：");
    scanf("%d",&n_folds);
    int k;
    double sum=0;
    double l_rate=0.5;
    int n_epoch=200;
    double *scores; 
    char filename[]="./sonar.csv";
    row=get_row(filename);
    col=get_col(filename);
    fold_size=(int)(row/n_folds);
    /*判断读取文件是否成功
    FILE *fp=NULL;
    fp = fopen(filename, "r");
    if(fp==NULL)
    {
        printf("WRONG!请检查路径！");
    }
    else
    {
        printf("%p\n",fp);
    }
    */
    char line[1024];
    double **data;//data要存放在二维数组里面，所以开双层指针
    data = (double **)malloc(row * sizeof(double *));
    //给二维数组data申请空间，data指向的指针数组需要n*sizeof(double*)，data[i]指向的指针数组需要n*sizeof(double)
    for (int i = 0; i < row; ++i){
        data[i] = (double *)malloc(col * sizeof(double));
    }//动态申请二维数组，data的每一个元素代表它的一行，每一行占的空间=sizeof(double)*col
    get_two_dimension(line, data, filename);
    printf("row = %d\n", row);
    printf("col = %d\n", col);
    int lendata=row;//求出数据集二维数组的长度
    //printf("LENDATA=%d\n",lendata);
    //print_two_dimension(data, row, col);
    //文件已经存到二维数组data中，R，M也进行了转换，data包装在结构体datas里面。
    scores=evaluate_algorithm(data,row);
    for(k=0;k<n_folds;k++)
    {
        printf("第%d次训练得分：",k+1);
        printf("%lf\n",scores[k]);
        sum+=scores[k];
    }
    double avgscore=sum/n_folds;
    printf("Average Score：%lf\n",avgscore);
    return 0;
}

void get_two_dimension(char *line, double **data, char *filename)
{
    FILE* stream = fopen(filename, "r");
    int i = 0;
    while (fgets(line, 1024, stream)!=NULL)//逐行读取,每次读取完一整行都把内容暂存到line[1024]中，下一次再读取一整行是line[1024]中会被放上新的一行。
    {
        int j = 0;
        char *tok;
        char* tmp = strdup(line);//strdup将字符串拷贝到新建的位置处，一般和free成对出现  
        for (tok = strtok(line, ","); tok && *tok; j++, tok = strtok(NULL, ",\n")){
            //for循环中：
            //（1）每次tok也随着j加一而向后挪一个逗号
            //（2）tok不为NULL代表没有读取完这一行的所有值，
            //    *tok不为NULL代表这一个tok指向的字符串的首位不为空，也就是说这个字符串总体不为空。
            /*最后一位作为分类位*/
            if(*tok=='R')
            {
                *(data[i]+j) = 1.0;
            }
            else if(*tok=='M')
            {
                *(data[i]+j) = 0.0;
            }
            else{   
            *(data[i]+j) = atof(tok);//转换成浮点数存入申请好的数组空间里。
            }
        }//字符串拆分操作
        
        i++;
        free(tmp);//释放内存空间
    }
    fclose(stream);//文件打开后要进行关闭操作
}
/*
void print_two_dimension(double** data, int row, int col)
{
    int i, j;
    for(i=1; i<row; i++){
        for(j=0; j<col; j++){
            printf("%f\t", *(data[i]+j));
        }
        printf("\n");
    }//打印的时候不打印第一行，第一行全是字符
}
*/

int get_row(char *filename)
{
    char line[1024];
    int i = 0;
    FILE* stream = fopen(filename, "r");
    while(fgets(line, 1024, stream)!=NULL){
        i++;
    }//读了多少行，就返回这个值
    fclose(stream);
    return i;
}

int get_col(char *filename)
{
    char line[1024];
    int i = 0;
    FILE* stream = fopen(filename, "r");
    fgets(line, 1024, stream);
    char* token = strtok(line, ",");
    while(token){
        token = strtok(NULL, ",");//token每次往后挨一个逗号
        i++;
    }
    fclose(stream);
    return i;//把第一行的列数算出来返回
}

//随机把所有数据拆分为n_folds个组，返回一个三维数组
double ***k_cross_validation(double **data, int lendata){
    /*获取全局变量的值*/
    srand(10);//种子
    char filename[]="./sonar.csv";
    row=get_row(filename);
    col=get_col(filename);
    fold_size=(int)(row/n_folds);

    printf("FOLD_SIZE:%d\n",fold_size);
    double ***split;
    int i,j=0,k=0;
    int index;
    double **fold;
    split=(double***)malloc(n_folds*sizeof(double**));
    for(i=0;i<n_folds;i++)
    {
        fold = (double**)malloc(fold_size*sizeof(double *));
        
        while(j<fold_size)
        {
            fold[j]=(double*)malloc(col*sizeof(double));
            index=rand()%lendata;
            fold[j]=data[index];
            //printf("%lf\n",fold[j][0]);//观察一下程序是否完整读取了数据
            for(k=index;k<lendata-1;k++)//for循环删除这个数组中被rand取到的元素
            {
                data[k]=data[k+1];
            }
            lendata--;//每次随机取出一个后总行数-1，保证不会重复取某一行
            j++;
        }
        j=0;//清零j
        split[i]=fold;
    }
    return split;
}

//模型准确率计算函数
double accuracy_metric(double* actual,double* predicted,int len_actual){
    int correct=0,i=0;
    for(i=0;i<len_actual;i++)
    {
        if(actual[i]==predicted[i])
            correct++;
    }
    return correct/(double)len_actual;
}

//n_folds是拆分成的组数。这个函数返回模型得分数组。
double *evaluate_algorithm(double **data, int lendata){
    double test;
    /*获取全局变量的值*/
    char filename[]="./sonar.csv";
    row=get_row(filename);
    col=get_col(filename);
    fold_size=(int)(row/n_folds);
    /*关键的数组重组部分*/
    double ***folds,***train_set,***kcross;//从k_cross_validation函数传过来的三维数组
    int i,j,k,l,m,n;//循环变量，i不能用！
    double *scores;//返回的模型得分数组
    scores=(double*)malloc(n_folds*sizeof(double));//scores的维数应该和三维数组的第一维相同，也就是训练集被分成的块数。
    kcross=k_cross_validation(data,lendata);
    double **final_train;//用来存放最终的训练集，二维数组
    double **final_test;//用来存放预测集，二维数组
    double *actual;//用来存放预测集的正确答案
    double *predicted;//算法预测生成的答案
    double accuracy;//单个准确度元素
    for(i=0;i<n_folds;i++)
    {//主循环
        /*内存分配：始*/
        final_train=(double**)malloc((n_folds-1)*fold_size*sizeof(double*));
        final_test=(double**)malloc(fold_size*sizeof(double*));
        actual=(double*)malloc(fold_size*sizeof(double));
        predicted=(double*)malloc(fold_size*sizeof(double));//actual和predicted有相同的维数，都是每块训练集的行数

        train_set=(double***)malloc((n_folds-1)*sizeof(double**));
        for(l=0;l<n_folds-1;l++)
        {
            train_set[l]=(double**)malloc(fold_size*sizeof(double*));
            for(m=0;m<fold_size;m++)
            {
                train_set[l][m]=(double*)malloc(col*sizeof(double));
            }
        }
        folds=(double***)malloc(n_folds*sizeof(double**));
        for(l=0;l<n_folds;l++)
        {
            folds[l]=(double**)malloc(fold_size*sizeof(double*));
            for(m=0;m<fold_size;m++)
            {
                folds[l][m]=(double*)malloc(col*sizeof(double));
            }
        }
        /*内存分配：终*/
        /*重置表达式：始*/
        for(j=0;j<n_folds;j++)
        {
            for(k=0;k<fold_size;k++)
            {
                for(l=0;l<col;l++)
                {
                    folds[j][k][l]=kcross[j][k][l];
                }   
            }
        }
        /*重置表达式：终*/
        

        final_test=folds[i];
        for(k=i;k<n_folds-1;k++)
        {
            folds[k]=folds[k+1];
        }
        
        for(j=0;j<n_folds-1;j++)
        {
            train_set[j]=folds[j];
        }
        for(j=0;j<(n_folds-1);j++)
        {
            for(l=0;l<fold_size;l++)
            {
                final_train[j*(fold_size)+l]=(double*)malloc(col*sizeof(double));
                final_train[j*(fold_size)+l]=train_set[j][l];                
            }
        }
        //final_train:((n_folds-1)*fold_size)*col
        for(n=0;n<fold_size;n++)
        {
            actual[n]=final_test[n][col-1];
            final_test[n][col-1]=-1.0;
        }
        predicted = perceptron(final_train,final_test,l_rate,n_epoch);
        accuracy=accuracy_metric(actual,predicted,fold_size);
        scores[i]=accuracy;
    }
    return scores;
}

/*以下是perceptron的专属部分*/
//整个算法过程中把bias放在weights的第一位，不再单独定义bias。
double predict(double *rows,double *weights){
    /*获取全局变量的值*/
    char filename[]="./sonar.csv";
    row=get_row(filename);
    col=get_col(filename);
    fold_size=(int)(row/n_folds);
    int i;
    double activation=weights[0];
    for(i=0;i<col-1;i++)//数组row的最后一位是分类结果，数组weights的第一位是bias。
    {
        activation+=weights[i+1]*rows[i];
    }
    if(activation>=0.0)
        return 1.0;
    else
        return 0.0;
}

double *train_weights(double **train, double l_rate, int n_epoch)
{
    /*获取全局变量的值*/
    char filename[]="./sonar.csv";
    row=get_row(filename);
    col=get_col(filename);
    fold_size=(int)(row/n_folds);
    //train是二维数组
    double prediction,error;
    int i,epoch,j,k;
    double *weights;
    int col=get_col(filename);
    weights=(double*)malloc(col*sizeof(double));
    for(i=0;i<col+1;i++)
    {
        weights[i]=0;
    }//其实这里用memset更方便
    for(epoch=0;epoch<n_epoch;epoch++)
    {
        for(j=0;j<(n_folds-1)*fold_size;j++)
        {
            prediction=predict(train[j],weights);
            error=train[j][col-1]-prediction;
            weights[0]=weights[0]+l_rate*error;//更新bias
            for(k=0;k<col-1;k++)
            {
                weights[k+1]=weights[k+1]+l_rate*error*train[j][k];//更新权重
            }
        }
    }
    return weights;
}

double *perceptron(double **train, double **test, double l_rate, int n_epoch)
{
    /*获取全局变量的值*/
    char filename[]="./sonar.csv";
    row=get_row(filename);
    col=get_col(filename);
    fold_size=(int)(row/n_folds);
    double *predictions;
    double prediction;
    double *weights;
    int i;
    weights=(double*)malloc(col*sizeof(double));//weights数组的长度就是列数（少一个结果位，多一个bias）
    predictions=(double*)malloc(fold_size*sizeof(double));//预测集的行数就是数组prediction的长度
    weights=train_weights(train,l_rate,n_epoch);
    for(i=0;i<fold_size;i++)
    {
        prediction=predict(test[i],weights);
        predictions[i]=prediction;
    }
    return predictions;
}







