#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

/*全局变量声明*/
static int n_folds;//为了避免麻烦，多个参数都要使用n_folds这一参数，直接设为全局变量。
static int row, col;//CSV文件的行、列数
static int fold_size;//fold_size是数据集被分成k块后每一块的长度
static double l_rate=0.001;
static int n_epoch=50;
//公共部分函数声明（CSV读取与交叉检验）
void get_two_dimension(char* line, double** data, char *filename);
double accuracy_metric(double* actual,double* predicted,int len_actual);
double *evaluate_algorithm(double **data, int lendata);
double ***k_cross_validation(double **data, int lendata);
int get_row(char *filename);
int get_col(char *filename);

//SLR函数声明
double predict(double *row, double *coef);
double *coef_sgd(double **train, double l_rate, int n_epoch);
double *linear_regression_sgd(double **train, double **test, double l_rate, int n_epoch);

//数据集预处理
double **dataset_minmax(double **dataset);
void normalize_dataset(double **dataset);

int main(){

    printf("请输入交叉检验拆分的组数：");
    scanf("%d",&n_folds);
    double sum=0.0;
    //printf("请输入交叉检验分成的块数：");
    //scanf("%d",&n_folds);
    char filename[]="./winequality-white.csv";
    row=get_row(filename);
    col=get_col(filename);
    fold_size=(int)(row/n_folds);
    printf("row = %d\n", row);
    printf("col = %d\n", col);
    double **data;
    data=(double**)malloc(row*sizeof(double*));
    for(int i=0;i<row;i++)
    {
        data[i]=(double*)malloc(col*sizeof(double));
    }
    //readcsv
    char line[1024];
    get_two_dimension(line, data, filename);
    normalize_dataset(data);
    double *scores;//均方误差
    scores=(double*)malloc(sizeof(double)*fold_size);

    scores=evaluate_algorithm(data,row);
    for(int k=0;k<n_folds;k++)
    {
        printf("第%d次训练均方误差：",k+1);
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
            //printf("看看：%s\n",tok);
            *(data[i]+j) = atof(tok);//转换成浮点数存入申请好的数组空间里。
        }//字符串拆分操作
        
        i++;
        free(tmp);//释放内存空间
    }
    fclose(stream);//文件打开后要进行关闭操作
}

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

double ***k_cross_validation(double **data, int lendata){
    /*获取全局变量的值*/
    srand(2);//种子
    char filename[]="./winequality-white.csv";
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
            //printf("RANDOM--INDEX:%d\n",index);
            //printf("RANDOM--INDEX:%lf\n",data[index][0]);
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
        //printf("第%d次循环\n",i+1);
    }
    return split;
}

//模型均方误差计算函数
double accuracy_metric(double* actual,double* predicted,int len_actual){
    double sumerr=0,avgerr;
    double error;
    for(int i=0;i<len_actual;i++)
    {
        error=predicted[i]-actual[i];
        sumerr+=pow(error,2);
    }
    avgerr=sumerr/len_actual;
    return sqrt(avgerr);
}

//传入perceptron算法函数的指针，做一个回调。n_folds是拆分成的组数。这个函数返回模型得分数组。
//struct array3{double ***a;};//三维数组放进结构体
double *evaluate_algorithm(double **data, int lendata){
    double test;
    /*获取全局变量的值*/
    char filename[]="./winequality-white.csv";
    row=get_row(filename);
    col=get_col(filename);
    fold_size=(int)(row/n_folds);

    /*关键的数组重组部分*/
    double ***folds,***train_set,***kcross;//从k_cross_validation函数传过来的三维数组
    int i,j,k,l,m,n;//循环变量，i不能用！
    double *scores;//返回的模型得分数组
    scores=(double*)malloc(n_folds*sizeof(double));//scores的维数应该和三维数组的第一维相同，也就是训练集被分成的块数。
    kcross=k_cross_validation(data,lendata);
    /*观察kcross是否正确
    for(int q=0;q<n_folds;q++)    
    {
        for(int w=0;w<fold_size;w++)
        {
            printf("%lf\n",kcross[q][w][0]);
        }
    }
    */
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
        //printf("foldxxxx:%lf\n",foldx.a[0][0][0]);

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
        predicted = linear_regression_sgd(final_train,final_test,l_rate,n_epoch);
        accuracy=accuracy_metric(actual,predicted,fold_size);
        //printf("ACCURACY:%lf\n",accuracy);
        scores[i]=accuracy;
    }
    return scores;
}

/*下面三个是专用函数*/
double predict(double *row, double *coef)
{
    double yhat;
    yhat=coef[0];
    for(int i=0;i<col-1;i++)
    {
        yhat+=coef[i+1]*row[i];//每一行前col-1个元素是自变量
    }
    return yhat;
}
//注意：coef第一个元素是截距，后面col-1个是每一列的相关系数
double *coef_sgd(double **train, double l_rate, int n_epoch)
{
    double *coef;
    double yhat,error;
    coef=(double*)malloc(col*sizeof(double));
    double sumerr;
    for(int i=0;i<col;i++)
    {
        coef[i]=0;
    }
    for(int epoch=0;epoch<n_epoch;epoch++)
    {
        sumerr=0;
        for(int j=0;j<(n_folds-1)*fold_size;j++)
        {
            yhat=predict(train[j],coef);
            error=yhat-train[j][col-1];
            sumerr+=pow(error,2);
            coef[0]=coef[0]-l_rate*error;//更新截距
            for(int k=0;k<col-1;k++)
            {
                coef[k+1]=coef[k+1]-l_rate*error*train[j][k];//更新相关系数
            }
        }
        //printf("epoch=%d,lrate=%lf,error=%lf\n",epoch,l_rate,sumerr);
    }
    return coef;
}

double *linear_regression_sgd(double **train, double **test, double l_rate, int n_epoch)
{
    double *predictions;
    double *coef;
    double yhat;
    predictions=(double*)malloc(sizeof(double)*fold_size);
    coef=(double*)malloc(sizeof(double)*col);
    coef=coef_sgd(train,l_rate,n_epoch);
    for(int i=0;i<fold_size;i++)
    {
        yhat=predict(test[i],coef);
        predictions[i]=yhat;
    }
    return predictions;
}

/*下面两个是数据集预处理函数*/
double **dataset_minmax(double **dataset){
    double **minmax;
    minmax=(double**)malloc(col*sizeof(double*));
    for(int i=0;i<col;i++)
    {
        minmax[i]=(double*)malloc(2*sizeof(double));
    }
    double *colvalues;//colvalues是一整列数，因此数组长度为row。
    colvalues=(double*)malloc(row*sizeof(double));
    
    double value_min;
    double value_max;
    for(int j=0;j<col;j++)
    {
        for(int i=0;i<row;i++)
        {
            colvalues[i]=dataset[i][j];
        }
        //value_min=min(colvalues)
        value_min=colvalues[0];
        for(int k=0;k<row;k++)
        {
            if(colvalues[k]<value_min)
                value_min=colvalues[k];
        }
        value_max=colvalues[0];
        for(int k=0;k<row;k++)
        {
            if(colvalues[k]>value_max)
                value_max=colvalues[k];
        }
        minmax[j][0]=value_min;
        minmax[j][1]=value_max;
    }
    return minmax;
}

void normalize_dataset(double **dataset){
    double **minmax;
    /*
    minmax=(double**)malloc(col*sizeof(double*));
    for(int i=0;i<col;i++)
    {
        minmax[i]=(double*)malloc(2*sizeof(double));
    }
    */
    minmax=dataset_minmax(dataset);
    for(int i=0;i<row;i++)
    {
        for(int j=0;j<col;j++)
        {
            dataset[i][j]=(dataset[i][j]-minmax[j][0])/(minmax[j][1]-minmax[j][0]);
        }
    }
}