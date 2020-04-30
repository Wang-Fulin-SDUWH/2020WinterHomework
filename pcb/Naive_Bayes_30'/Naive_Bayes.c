#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#define pi 3.141593

struct dictionary{
    int key;
    double **arr;
};
struct dict_len{
    int *lenkey;//长度数组
    struct dictionary *sep;//结构体dict构成的数组
};
struct summary{
    int key;
    double **summary;
};


/*全局变量声明*/
static int classnum=3;//要分成几类（鸢尾花数据集分三类）
static int n_folds;//为了避免麻烦，多个参数都要使用n_folds这一参数，直接设为全局变量。
static int row, col;//CSV文件的行、列数
static int fold_size;//fold_size是数据集被分成k块后每一块的长度

//公共部分函数声明（CSV读取与交叉检验）
void get_two_dimension(char* line, double** data, char *filename);
void print_two_dimension(double** data, int row, int col);
double accuracy_metric(double* actual,double* predicted,int len_actual);
double *evaluate_algorithm(double **data, int lendata);
double ***k_cross_validation(double **data, int lendata);
int get_row(char *filename);
int get_col(char *filename);

//Naive_Bayes函数声明
struct dict_len separate_by_class(double **dataset,int lendata);
struct summary *summarize_by_class(double** dataset,int lendata);
double mean(double *numbers, int len);
double stdev(double *numbers, int len);
double calculate_probability(double x,double mean,double stdev);
int calculate_class_probabilities(struct summary *sum, double *test);
double *naive_Bayes(double **train, double **test);


int main(){
    double sum=0.0;
    printf("请输入交叉检验分成的块数：");
    scanf("%d",&n_folds);
    char filename[]="./iris.csv";
    row=get_row(filename);
    col=get_col(filename);
    fold_size=(int)(row/n_folds);
    classnum=3;
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

    //data是读取的csv文件

/*
    struct dict_len sep;
    sep=separate_by_class(data,row);
    for(int j=0;j<classnum;j++)
    {
        printf("KEY：%d\n",sep.sep[j].key);
        for(int k=0;k<sep.lenkey[j];k++)
        {
            printf("ELEMENTS:%lf\n",sep.sep[j].arr[k][0]);
        }
    }
    struct summary *sum;
    
    sum=summarize_by_class(data,row);

    for(int i=0;i<classnum;i++)
    {
        printf("CLASS_KEY:%d\n",sum[i].key);
        for(int k=0;k<col-1;k++)
        {
            printf("%lf\n",sum[i].summary[k][0]);
            printf("%lf\n",sum[i].summary[k][1]);
            printf("\n");
        }
        printf("\n");
    }
*/
    double *scores;//得分
    //scores=(double*)malloc(sizeof(double)*fold_size);
    scores=evaluate_algorithm(data,row);
    for(int k=0;k<n_folds;k++)
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
    char a[]="Iris-setosa";
    char b[]="Iris-versicolor";
    char c[]="Iris-virginica";
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
            if(strcmp(tok,a)==0)
            {
                *(data[i]+j)=0.0;
            }
            else if(strcmp(tok,b)==0)
            {
                *(data[i]+j)=1.0;
            }
            else if(strcmp(tok,c)==0)
            {
                *(data[i]+j)=2.0;
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
    srand(10);//种子
    char filename[]="./iris.csv";
    row=get_row(filename);
    col=get_col(filename);
    fold_size=(int)(row/n_folds);
    classnum=3;

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

//传入perceptron算法函数的指针，做一个回调。n_folds是拆分成的组数。这个函数返回模型得分数组。
//struct array3{double ***a;};//三维数组放进结构体
double *evaluate_algorithm(double **data, int lendata){
    double test;
    /*获取全局变量的值*/
    char filename[]="./iris.csv";
    row=get_row(filename);
    col=get_col(filename);
    fold_size=(int)(row/n_folds);
    classnum=3;
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
        predicted = naive_Bayes(final_train,final_test);
        accuracy=accuracy_metric(actual,predicted,fold_size);
        //printf("ACCURACY:%lf\n",accuracy);
        scores[i]=accuracy;
    }
    return scores;
}


struct dict_len separate_by_class(double **dataset, int lendata)//数据集拆分成“字典”的函数。返回一个结构体，里面有“字典”以及字典中每个value对应的二维数组的长度。
{
    //全局变量
    char filename[]="./iris.csv";
    row=get_row(filename);
    col=get_col(filename);
    fold_size=(int)(row/n_folds);
    classnum=3;
    //考虑开一个row*sizeof(double*)的空间来装每个key对应的巨大的二维数组。（二维数组的dim1不会超过总行数row）
    int i,j,k,l;
    double *vector;
    int *lenarr;
    int class_value;
    vector=(double*)malloc(col*sizeof(double));
    struct dictionary *separated;
    struct dict_len sep;
    separated=(struct dictionary*)malloc(classnum*sizeof(struct dictionary));
    lenarr=(int*)malloc(classnum*sizeof(int));
    
    //初始化
    for(k=0;k<classnum;k++)
    {
        lenarr[k]=0;//初始化lenarr数组为0，意味着在没有某一行被放入结构体时，长度为0.
        separated[k].key=-1;//避免出错，把结构体数组中所有结构体的key都设置成1。
    }
    int count=0,flag=0;//countarr是每个arr中的元素个数，count是结构体数组中的结构体个数。
    //每个arr元素个数并不一样，因此count每自增1，就在和count相关的长度数组中增加一个元素。
    //Eg:结构体数组长这样:{{0,[[arr1],[arr2],....]},{1,[[arr1],[arr2],......]},{2,[[]]},......}
    //对应的开一个数组lenarr:{lenarr1,lenarr2,lenarr3,......}
    /*这段内存分配有些多余了。应该直接去flag==0里面内存第一次分配，之后每增加一个realloc一次。*/
    for(j=0;j<classnum;j++)
    {
        separated[j].arr=(double**)malloc(lenarr[j]*sizeof(double*));
    }
    
    for(i=0;i<lendata;i++)
    {
        vector=dataset[i];
        class_value=vector[col-1];
        for(j=0;j<classnum;j++)
        {
            if(separated[j].key==class_value)//如果不需要新的key来存放class_value，那么在检索到的key中把vector放进separated[j].arr中。
            //这也会使lenarr[j]自增1。
            {
                flag=1;//伪代码separated[j].arr.append(vector);
                separated[j].arr=(double**)realloc(separated[j].arr,(lenarr[j]+1)*sizeof(double*));//重新分配更大的内存（不行就干脆上面全改成row*sizeof(double*)）
                separated[j].arr[lenarr[j]]=vector;//新分配的内存用来存放vector
                lenarr[j]++;
                break;
            }
        }
        if(flag==0)//原来没有对应的key来存放这个vector
        {
            separated[count].key=class_value;
            //伪代码separated[count].arr.append(vector);
            separated[count].arr=(double**)realloc(separated[count].arr,sizeof(double*));//新加入第一个的时候，分内存分一个就可以了。
            separated[count].arr[lenarr[count]]=vector;
            lenarr[count]++;
            count++;//下标+1，下次再来新的key就往新的下标里面加key。
        }
        flag=0;   
    }
    sep.lenkey=lenarr;
    sep.sep=separated;
    return sep;
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

double stdev(double *numbers, int len){
    double avg,var=0;
    int i;
    avg=mean(numbers,len);
    for(i=0;i<len;i++)
    {
        var+=pow((numbers[i]-avg),2);
    }
    var=var/(len-1);
    return sqrt(var);
}

//目标生成的数据概述: {0: [(2.7419599999999997, 0.926544981638776, 5), (3.0054, 1.1073196467145339, 5)], 
//                  1: [(7.6146, 1.2344272457297758, 5), (2.9914199999999997, 1.454185609542331, 5)]}

//Eg:{0: [(2.7419599999999997, 0.926544981638776, 5), (3.0054, 1.1073196467145339, 5)],
// 1: [(7.446833333333333, 1.713670167603245, 3), (3.5804999999999993, 1.0934449963304054, 3)], 
// 2: [(7.86625, 0.10401540751254115, 2), (2.1078, 1.8613878907954677, 2)]}

//[mean,stdev,len(col)]
struct summary *summarize_by_class(double** dataset,int lendata){
    /*全局变量*/
    char filename[]="./iris.csv";
    row=get_row(filename);
    col=get_col(filename);
    fold_size=(int)(row/n_folds);
    classnum=3;

    struct dict_len separated = separate_by_class(dataset,lendata);
    struct summary *sum;//要返回一个数据概述sum
    sum=(struct summary*)malloc(classnum*sizeof(struct summary));
    for(int i=0;i<classnum;i++)
    {
        sum[i].summary=(double**)malloc(col*sizeof(double*));
        for(int k=0;k<col;k++)
        {
            sum[i].summary[k]=(double*)malloc(3*sizeof(double));
        }//summary是要返回的总结二维数组，对每一列做概述（mean,std,行数）。
    }
    //sum的形式:{{0,[[第一列的三个特征],[第二列的三个特征],...]},{1,[[第一列的三个特征],[],...]},{2,[[],[],...]},...}

    //separated.lenkey是结构体数组中每个元素的value表示的二维数组的长度,separated.sep是结构体”字典“数组。
    //注意整个大的结构体数组的长度就是classnum。所有数据要被分成几类在写代码之前肯定是知道的吧.......
    //以下的结构示意中为了区分，把数组记作[]。
    //separated.sep: [{0,[[a1,a2,...,acol,0],[b1,b2,...,0],[],...]},{1,[[],[],[],...]},{2,[[],[],[],...]}]
    double ***temp;//temp是违规操作吗？
    temp=(double***)malloc(classnum*sizeof(double**));
    //开一个三维数组存放每一列。为了方便，最后一列就先留着。之后取用的时候再做调整。
    for(int j=0;j<classnum;j++)
    {
        temp[j]=(double**)malloc((col)*sizeof(double*));
    //separated.sep[j].arr是要被拿出每一列的第j个二维数组。separated.lenkey[j]是第j个二维数组的行数。
        for(int k=0;k<col;k++)//列
        {
            temp[j][k]=(double*)malloc(separated.lenkey[j]*sizeof(double));
            for(int l=0;l<separated.lenkey[j];l++)//行
            {
                temp[j][k][l]=separated.sep[j].arr[l][k];//m*n的二维数组取出每一列放到n*m的二维数组里面
            }
        }
    //现在temp[j]里面的每个元素(temp[j][k])是第j个class所对应的二维数组的第k列。它的长度是separated.lenkey[j]。
    }
    for(int j=0;j<classnum;j++)
    {
        sum[j].key=separated.sep[j].key;
        for(int k=0;k<col;k++)
        {
            sum[j].summary[k][0]=mean(temp[j][k],separated.lenkey[j]);
            sum[j].summary[k][1]=stdev(temp[j][k],separated.lenkey[j]);
            sum[j].summary[k][2]=separated.lenkey[j];
        }
    }
    return sum;
}

double calculate_probability(double x,double mean,double stdev){
    double exponent;
    exponent=exp(-(pow(x-mean,2)/(2*pow(stdev,2))));
    return exponent/(sqrt(2*pi)*stdev);
}
//举例：p(嫁|不帅，性格不好，不上进，身高矮)
//p(选择类别n|样本特征)=p(类别n|样本特征1，样本特征2，...，样本特征m)
//=p(样本特征1，样本特征2，...，样本特征m|类别n)*p(类别n)/分母
//分母都是一样的。
//因此只需求p(样本特征1|类别n)*p(样本特征2|类别n)*...*p(样本特征m|类别n)*p(类别n)
//calculate_probability函数的使用：
//输入待预测样本的第k维数据，第l类的训练集第k列数据的均值、标准差。得到p(样本特征k|类别l)
//struct summary：
//{0: [(2.7419599999999997, 0.926544981638776, 5), (3.0054, 1.1073196467145339, 5)],
// 1: [(7.446833333333333, 1.713670167603245, 3), (3.5804999999999993, 1.0934449963304054, 3)], 
// 2: [(7.86625, 0.10401540751254115, 2), (2.1078, 1.8613878907954677, 2)]}

int calculate_class_probabilities(struct summary *sum, double *test)
//参数说明：sum是summary结构体，包含着各类样本每一列的相关数据（每一类样本对应二维数组的行都为col，列都为3）
//test是二维数组dataset的某一行。
//pr_class是用来存放概率的数组，一共有classnum个元素。返回pr_class。
{
    char filename[]="./iris.csv";
    row=get_row(filename);
    col=get_col(filename);
    fold_size=(int)(row/n_folds);
    classnum=3;

    double *pr_class;
    int count=0;//临时下标
    pr_class=(double*)malloc(classnum*sizeof(double));
    for(int i=0;i<classnum;i++)
    {
        pr_class[i]=sum[i].summary[0][2]/row;
        //这一类的样本数/总行数。sum[i].summary[0][2]和上面的separated.lenkey[i]是一样的。
        //这样避免了过多的传参麻烦。
    }//初始化概率数组为p(类别n)（下面要*=）
    for(int j=0;j<classnum;j++)
    {
        for(int k=0;k<col-1;k++)
        {
            pr_class[j]*=calculate_probability(test[k],sum[j].summary[k][0],sum[j].summary[k][1]);
        }
    }
    for(int i=0;i<classnum;i++)
    {
        //printf("属于%d类的概率:",sum[i].key);
        //printf("%lf\n",pr_class[i]);
        if(pr_class[count]<pr_class[i])
            count=i;
    }
    return sum[count].key;//返回下标
}

double *naive_Bayes(double **train, double **test){
    double *predict;
    predict=(double*)malloc(sizeof(double)*fold_size);
    char filename[]="./iris.csv";
    row=get_row(filename);
    col=get_col(filename);
    fold_size=(int)(row/n_folds);
    classnum=3;
    struct summary *sum=summarize_by_class(train,(n_folds-1)*fold_size);
    for(int i=0;i<fold_size;i++)
    {
        predict[i]=calculate_class_probabilities(sum,test[i]);
    }
    return predict;
}


/*测试数据
    double **data;
    data=(double**)malloc(row*sizeof(double*));
    for(int i=0;i<row;i++)
    {
        data[i]=(double*)malloc(col*sizeof(double));
    }
    data[0][0]=3.3935;
    data[0][1]=2.3312;
    data[0][2]=0;
    data[1][0]=3.1100;
    data[1][1]=2.3312;
    data[1][2]=0;
    data[2][0]=1.3438;
    data[2][1]=3.3683;
    data[2][2]=0;
    data[3][0]=7.9398;
    data[3][1]=0.7916;
    data[3][2]=1;
    data[4][0]=3.5822;
    data[4][1]=4.6791;
    data[4][2]=0;
    data[5][0]=2.2803;
    data[5][1]=2.8669;
    data[5][2]=0;
    data[6][0]=7.4234;
    data[6][1]=4.6965;
    data[6][2]=1;
    data[7][0]=5.7450;
    data[7][1]=3.5339;
    data[7][2]=1;
    data[8][0]=9.1721;
    data[8][1]=2.5111;
    data[8][2]=1;
    data[9][0]=7.7927;
    data[9][1]=3.4240;
    data[9][2]=1;
    */