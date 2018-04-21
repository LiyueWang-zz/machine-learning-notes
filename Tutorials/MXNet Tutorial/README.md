
# MXNet Tutorial
个人笔记。

[中文教程](http://zh.gluon.ai/)

[第一季课程视频](https://discuss.gluon.ai/t/topic/753)

[课程源代码](https://github.com/mli/gluon-tutorials-zh)

## 目录

* [MXNet Tutorial](#mxnet-tutorial)
    * [目录](#目录)
	* [预备知识](#预备知识)
	    * [机器学习简介](#机器学习简介)
		    * 机器学习四要素(#机器学习四要素)
			* 监督学习(#监督学习——supervised-learning)
			* 无监督学习(#无监督学习——unsupervised-learning)
			* 与环境因素交互(#与环境因素交互——Online Learning)
		* 使用NDArray来处理数据(#使用ndarray来处理数据)
		* 使用autograd来自动求导(#使用autograd来自动求导)
    * [深度学习模型基础](#深度学习模型基础)

# 预备知识

## 机器学习简介

### 机器学习四要素：
  * 数据(Data)
  * 模型(Models)
  * 损失函数(Loss Functions): 训练误差(training error), 测试误差(test error)
  * 优化算法(Optimization Algorithms): i.e. 梯度下降法
  
### 监督学习——Supervised Learning
training set with ground-truth label + supervised learning algorithm => learned model

* 回归分析(Regression): 输入时任意离散或连续的、单一或多个的变量，输出是连续的数值（某个范围内的任意实数值）。
* 分类(Classification): 输出是离散的类别。分类问题的损失函数-交叉熵(cross-entry)
* 搜索与排序(Search and Ranking): 从较大集合中生成一个有序子集。i.e. PageRank
* 推荐系统(Recommender Systems): 向用户展示一组相关条目。i.e. 搜索引擎的搜索条目自动补全系统
* 序列学习(Sequence Learning): 处理任意长度的输入序列，或输出任意长度的序列（或两者兼顾）。
  * 词类标注和句法分析(Tagging and Parsing) 
  * 语音识别(Automatic Soeech Recognition)
  * 文本转语音(Text to Speech)
  * 机器翻译(Machine Translation)

### 无监督学习——Unsupervised Learning

* 聚类(clustering): 
* 子空间估计(subspace estimation): 
* 主成分分析(principal component analysis)
* 表征学习(representation learning)
* 贝叶斯图模型(Probabilistic/Bayes Graphical Model)
* 生成对抗网络(generative adversarial networks)

### 与环境因素交互——Online Learning

* 强化学习(Reinforcement Learning): 机器人程序、对话系统、电子游戏AI. i.e.: AlphaGo, Deep Q Network
  * 马尔科夫决策过程(Markov Decision Process)
  * 情景式赌博机问题(Contextual bandit problem)
* 对抗学习(Adversarial learning)

## 使用NDArray来处理数据
[NDArray API](https://mxnet.incubator.apache.org/api/python/ndarray/ndarray.html)

## 使用autograd来自动求导
[link](http://zh.gluon.ai/chapter_crashcourse/autograd.html)


# 深度学习模型基础
