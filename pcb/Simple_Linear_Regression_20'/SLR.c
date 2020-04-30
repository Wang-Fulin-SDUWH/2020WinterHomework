#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#define pi 3.141593

struct tt{
    int train_size;
    int test_size;
    double **train;
    double **test;
};

/*全局变量声明*/
static double splits;
static int row, col;//CSV文件的行、列数

//公共部分函数声明（CSV读取与七三分检验）
void get_two_dimension(char* line, double** data, char *filename);
void print_two_dimension(double** data, int row, int col);
double accuracy_metric(double* actual,double* predicted,int len_actual);
double evaluate_algorithm(double **data,double split);//这里不是交叉检验，所以返回一个得分即可，不用返回很多个得分然后取平均
int get_row(char *filename);
int get_col(char *filename);

//简单线性回归函数声明
struct tt train_test_split(double **dataset, double split);
double mean(double *numbers, int len);
double variance(double *numbers, int len);
double covariance(double *x,double *y,int lenx);
double *coefficients(double **dataset,int len);
double *simple_linear_regression(double **train,double **test,int train_size,int test_size);


int main(){
    double sum=0.0;
    int test_size;
    printf("请输入训练集所占的比例：");
    scanf("%lf",&splits);
    char filename[]="./insurance.csv";
    row=get_row(filename);
    col=get_col(filename);
    test_size=row-(int)(row*splits);
    printf("row = %d\n", row);
    printf("col = %d\n", col);

    double **data;
    data=(double**)malloc(row*sizeof(double*));
    for(int i=0;i<row;i++)
    {
        data[i]=(double*)malloc(col*sizeof(double));
    }

    char line[1024];
    get_two_dimension(line, data, filename);

    double scores;//得分
    scores=evaluate_algorithm(data,splits);
    printf("最终均方误差：%lf\n",scores);
    if(scores<=72){
        printf("Better than Baseline of 72!\n");
    }
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


struct tt train_test_split(double **dataset, double split)
{
    srand(10);
    struct tt tt;
    int train_size,test_size,index;
    int lendata=row;
    train_size=(int)(row*split);
    test_size=row-train_size;
    /*内存分配：起*/
    double **train_set;
    double **test_set;
    train_set=(double**)malloc(train_size*sizeof(double*));
    for(int i=0;i<train_size;i++)
    {
        train_set[i]=(double*)malloc(col*sizeof(double));
    }
    test_set=(double**)malloc(test_size*sizeof(double*));
    for(int i=0;i<test_size;i++)
    {
        test_set[i]=(double*)malloc(col*sizeof(double));   
    }
    /*内存分配：止*/
    for(int i=0;i<train_size;i++)
    {
        index=rand()%lendata;
        //意图：train_set[i]=dataset[index];
        for(int j=0;j<col;j++)
        {
            train_set[i][j]=dataset[index][j];   
        }
        for(int k=index;k<lendata-1;k++)
        {
            //意图：dataset[k]=dataset[k+1];
            for(int l=0;l<col;l++)
            {
                dataset[k][l]=dataset[k+1][l];
            }
        }
        lendata--;
    }
    for(int i=0;i<test_size;i++)
    {
        //意图：test_set[i]=dataset[i]
        for(int j=0;j<col;j++)
        {
            test_set[i][j]=dataset[i][j];
        }  
    }
    tt.train=train_set;
    tt.test=test_set;
    tt.train_size=train_size;
    tt.test_size=test_size;
    return tt;
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

double evaluate_algorithm(double **data, double split){
    double **train;
    double **test;
    double rmse;
    struct tt spl;
    int train_size,test_size;
    spl=train_test_split(data,split);
    train=spl.train;
    test=spl.test;
    train_size=spl.train_size;
    test_size=spl.test_size;
    double **test0;//挖空最后一列的测试集
    /*给测试集分配内存*/
    test0=(double**)malloc(test_size*sizeof(double*));
    for(int i=0;i<test_size;i++)
    {
        test0[i]=(double*)malloc(col*sizeof(double));   
    }
    for(int k=0;k<test_size;k++)
    {
        for(int l=0;l<col-1;l++)
        {
            test0[k][l]=test[k][l];
        }
        test0[k][col-1]=-1;//挖空最后一位
    }
    double *predicted;
    double *actual;
    predicted=(double*)malloc(sizeof(double)*test_size);
    actual=(double*)malloc(sizeof(double)*test_size);
    for(int i=0;i<test_size;i++)
    {
        actual[i]=data[i][col-1];   
    }
    predicted=simple_linear_regression(train,test,train_size,test_size);
    rmse=accuracy_metric(actual,predicted,test_size);
    return rmse;
}

double mean(double *numbers, int len){
    int i;
    double sum=0,avg;
    for(i=0;i<len;i++)
    {
        sum+=numbers[i];
    }
    avg=sum/len;
    return avg;
}

double variance(double *numbers, int len){
    double avg,var=0;
    int i;
    avg=mean(numbers,len);
    for(i=0;i<len;i++)
    {
        var+=pow((numbers[i]-avg),2);
    }
    var=var/(len-1);
    return var;
}

double covariance(double *x,double *y,int lenx)
{
    double covar=0.0;
    for(int i=0;i<lenx;i++)
    {
        covar+=(x[i]-mean(x,lenx)*(y[i]-mean(y,lenx)));
    }
    return covar;
}

double *coefficients(double **dataset,int len)
{
    double *coe;
    coe=(double*)malloc(sizeof(double)*2);
    double *x,*y;
    double b1,b0;
    x=(double*)malloc(sizeof(double)*len);
    y=(double*)malloc(sizeof(double)*len);
    for(int i=0;i<len;i++)
    {
        x[i]=dataset[i][0];
        y[i]=dataset[i][1];
    }
    b1=covariance(x,y,len)/variance(x,len);
    b0=mean(y,len)-b1*mean(x,len);
    coe[0]=b0;
    coe[1]=b1;
    return coe;
}

double *simple_linear_regression(double **train,double **test,int train_size,int test_size)
{
    double *predictions;
    double yhat;
    predictions=(double*)malloc(sizeof(double)*test_size);
    double *coe;
    coe=coefficients(train,train_size);
    double b0,b1;
    b0=coe[0];
    b1=coe[1];
    for(int i=0;i<test_size;i++)
    {
        yhat=b0+b1*test[i][0];
        predictions[i]=yhat;
    }
    return predictions;
}