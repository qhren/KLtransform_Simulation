# KLtransform_Simulation
## 原理简介
KLtanrsform_Simulation实现的是一个对类似图片进行降维重建的过程，主要利用相似图片之间的相关性 <br>
来减少图片传输过程中的成本。<br>
### Step
1.计算输入的相似图片之间的协方差矩阵Cx <br>
2.对协方差矩阵进行eig操作，求出特征值和特征向量 <br>
3.对特征值进行排序，选择特征值较大的几个特征向量 <br>
4.归一化，计算KL变换的正变换和反变换矩阵 <br>
5.对输入的每一张图片，进行KL反变换，比较和原图片的视觉效果，以及误差。<br> 
#### 注
KL变换利用的是协方差矩阵，实际上，也可以利用类似的矩阵(比如相关矩阵等)，KL变换和PCA(主成分分析) <br>
其实是一种方法，KL变换还可以实现简单的人脸识别。 <br>
同样，我们也可以利用PCA的方法来实现对图像的降维重建<br>
源代码及注释已经包含在PCA_transform.m文件中 <br>
## PCA实现简单的**人脸识别**
利用训练的人脸数据集构成特征空间<br>
输入的图片投影到特征空间，进行（**欧氏**）距离度量<br>
这里有两个阈值，分别是：
1.训练集构成的特征空间的阈值，用于判定测试的人脸是都属于已知的人脸空间
2.训练集中，对于每一个人，都选取了几张图片进行训练，对于不同的人,<br>
也存在一个阈值，用于判定输入的测试图片属于哪个人<br>
3.判定过程为先判定是否属于训练集的特征空间，再按照距离（类似于*K近邻法*)<br>
具体的论文地址见：<br>
[PCA实现简单的人脸识别](https://wenku.baidu.com/view/e4c18db465ce0508763213be.html)
