# Speaker Diarization使用说明
### 1. 系统介绍
#### 1.1 功能
该脚本的功能是给定一段音频,可以将该音频中每一句话按照说话人进行聚类.
#### 1.2 原理
该系统前段采用x-vector extractor网络进行前端的处理,然后使用Joint Bayesian对每一个句子进行打分,最后使用层次聚类来得到每一个句子对应的说话人类别.
### 2. 数据集
数据集一共分为x-vector extractor的训练集,Joint Bayesian参数的训练集,开发集和测试集.

其中,训练集和测试集有可能属于不同的domain(例如英语数据集和汉语数据集).这时我们就需要找出一个和测试集语种相近,并且有标注的数据集作为开发集.开发集有两种功能.第一,我们利用开发集数据来对训练集数据进行adaptation,使训练集所属的domain更加接近测试集所属的domain.第二,与传统意义上的开发集的功能相同.在层次聚类环节,我们需要选取一个阈值来为测试集进行说话人聚类.在不知道测试集说话人标签的前提下,我们可以选取一个在开发集上表现最好的阈值应用在测试集上.
#### 2.1 x-vector extractor训练集
该训练集由SRE和SWBD数据集组成,并使用RIRS和musan进行加噪.该数据集一共有192793个音频文件,包含6399个说话人.加噪后总时长为8817小时.
#### 2.2 Joint Bayesian参数的训练集
该训练集在SRE数据集中选取了120000个句子进行训练.该部分数据集一共有3787个说话人,120000个句子,总时长为100个小时.
#### 2.3 开发集和测试集
在本程序中,我们选取Hub4m97数据集来做测试.该数据集一共有58个文件,包含了998个说话人,35310个句子,总时长为29个小时.在我们的程序中,需要用到和测试集属于同一个domain的其它数据集作为开发集.所以我们将Hub4m97数据集分成两个子数据集,一个作为开发集,一个作为测试集.开发集有29个文件,包含了606个说话人,14894句话,总时长为12.5个小时;测试集有29个文件,包含了392个说话人,20416句话,总时长为12.5个小时.


### 3. 使用方法

#### 3.1 训练x-vector extractor
```
python train_DNN.sh
```
#### 3.2 训练Joint Bayesian参数
```
python train_JB.sh
```
#### 3.3 对测试集打分并聚类
在有开发集的情况下：
```
python test_JB.sh
```
在没有开发集的情况下：
```
python test_JB_v2.sh
```
### 4. 常见错误及解决办法
#### 4.1 Output of qsub was: sh: 1: qsub: not found
打开cmd.sh.将
```
export train_cmd="queue.pl"
```
修改为
```
export train_cmd="run.pl"
```
#### 4.2 缺少某些文件
该错误的原因是train_DNN.sh,train_JB.sh和test_JB.sh这三个脚本的stage设置出现问题,导致跳步.根据报错信息,重新运行出错部分的上一个stage.
### 5. 文件清单
train_DNN.sh: 训练x-vector extractor的顶层脚本

train_JB.sh: 训练Joint Bayesian参数的顶层脚本

test_JB.sh: 对测试集打分并聚类的顶层脚本

test_JB_v2.sh: 在没有开发集的情况下对测试集打分并聚类的顶层脚本

data: 数据集的kaldi格式文件

dataloader/make_Hub4m97.py: 将Hub4m97数据集转化为kaldi格式

dataloader/make_Deliver.py: 将Deliver数据集转化为kaldi格式

JB: Joint Bayesian训练,测试等脚本

&emsp;&emsp;train.py: JB训练顶层文件

&emsp;&emsp;Feat.py: JB训练时数据集的读取

&emsp;&emsp;JB_train.py: JB训练类,包含了训练时使用的EM算法以及生成JB训练的参数

&emsp;&emsp;test.py: 测试集的JB评分顶层文件

&emsp;&emsp;sub_file.pl: 在整个的测试集中将某一个特定文件的所有x-vector单独分离出来  

&emsp;&emsp;JB_train.py: JB测试类,对两个x-vector计算似然比

&emsp;&emsp;PCA.py: 特征降维的函数

