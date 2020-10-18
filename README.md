# README

​        基础版本实现。

​        实现方案，用cnn提取文本特征向量，然后用全连接层得到最终结果，损失函数用均方损失函数。

### 一、install

安装anaconda3, 并加入path
conda create --name work python=3.7
conda activate work
conda install pytorch -c pytorch torchvision
conda install jieba
set PATH=%PATH%;%SystemRoot%\system32;%SystemRoot%;%SystemRoot%\System32\Wbem;
conda install matplotlib -c conda-forge

如果用的是pycharm，则要修改解释器
file -> settings -> Project -> Python Interpreter
修改解释器路径：PATH_TO_ANACONDA/envs/work/python.exe
例如我的路径是：D:\Usr\Anaconda\envs\work\python.exe

## 二、数据集下载

在data文件夹中，并将官网下载的数据集train.query.tsv train.reply.tsv放进去

## 三、运行

运行main.py即可（首次运行要先运行utils/txt2json生成中间数据）

## 四、TODO

更换模型，如svm、其它神经网络等。。。

更换损失函数？

词表划分可以人工调整一下（data/common_dict.txt中添加常用词）

没有划分训练集和验证集

没有转换成数据提交格式