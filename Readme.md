# Speaker Diarization使用说明
### 1. 系统介绍
#### 1.1 功能
该脚本的功能是给定一段音频,可以将该音频中每一句话按照说话人进行聚类.
#### 1.2 原理
该系统采用x-vector extractor网络进行前端的处理,之后使用Joint Bayesian对所有x-vector之间两两进行打分,最后使用层次聚类来得到每一个句子对应的说话人类别.
### 2. 数据集
数据集一共分为x-vector extractor的训练集,Joint Bayesian参数的训练集,Speaker Diarization的开发集和测试集.

其中,Joint Bayesian参数训练集和Speaker Diarization测试集有可能属于不同的domain(例如英语数据集和汉语数据集).这时我们就需要找出一个和测试集语种相近,并且有标注的数据集作为开发集.开发集有两种功能. 与传统意义上的开发集的功能相同.在层次聚类环节,我们需要选取一个阈值来为测试集进行说话人聚类.在不知道测试集说话人标签的前提下,我们可以选取一个在开发集上表现最好的阈值应用在测试集上.
#### 2.1 x-vector extractor训练集
该训练集由SRE和SWBD数据集组成,并使用RIRS和musan进行加噪.该数据集一共有192793个音频文件,包含6399个说话人.加噪后总时长为8817小时.
#### 2.2 Joint Bayesian参数的训练集
该训练集在SRE数据集中选取了120000个句子进行训练.该部分数据集一共有3787个说话人,120000个句子,总时长为100个小时.
#### 2.3 开发集和测试集
在本程序中,我们选取callhome数据集和Hub4m97数据集作为Speaker Diarization的开发集和测试集.callhome被分成两个子数据集:callhome1和callhome2,一个作为开发集,一个作为测试集.Hub4m97数据集分成两个子数据集Hub4m97_1和Hub4m97_2,一个作为开发集,一个作为测试集.

#### 2.4 数据集清单

|     数据集     | 音频文件数目 | 说话人数目 | 句子数目 | 音频时长(h) |
| :------------: | :----------: | :--------: | :------: | :---------: |
| x-vector训练集 |    192793    |    6399    |    -     |    8817     |
|    JB训练集    |      -       |    3787    |  120000  |     100     |
|    callhome    |     500      |    1286    |  29988   |    15.5     |
|   callhome1    |     250      |    645     |  15023   |      8      |
|   callhome2    |     250      |    641     |  14965   |     7.5     |
|    Hub4m97     |      58      |    998     |  35310   |     29      |
|   Hub4m97_1    |      29      |    606     |  14894   |    12.5     |
|   Hub4m97_2    |      29      |    392     |  20416   |    16.5     |
* 注: x-vector训练集不需要把音频按照句子进行切分,故没有统计句子数目;JB训练集是在SRE数据集中随机选取了120000个句子进行训练,故无法统计音频文件数目.


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


### 4. 结果

|  数据集  | 说话人未知的DER | 说话人已知的DER |
| :------: | :-------------: | :-------------: |
| callhome |      7.64%      |      6.51%      |
| Hub4m97  |     36.92%      |     37.82%      |


### 5. 常见错误及解决办法

#### 5.1 Output of qsub was: sh: 1: qsub: not found
打开cmd.sh.将
```
export train_cmd="queue.pl"
```
修改为
```
export train_cmd="run.pl"
```
#### 5.2 缺少某些文件
该错误的原因是train_DNN.sh,train_JB.sh和test_JB.sh这三个脚本的stage设置出现问题,导致跳步.根据报错信息,重新运行出错部分的上一个stage.
### 6. 文件清单
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

