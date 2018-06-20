function [ ]=KLtransform(K)
%% 对ORL_32*32中的数据进行图像降维重建，同时选取不同的K值来测试实验结果
% Background introduction
% 本次实验从数据集中取一个人的10张图片进行图像的降维重建
% 输入arg k为一个向量
%% 
load('ORL_32x32.mat');% 载入数据
for i=1:10
    source_image(:,i)=fea(i,:);
end
%将总体的实验样本以5*6的形式显示
for i=1:2
    for j=1:5
        show_image((1:32)+(j-1)*32,(1:32)+(i-1)*32)=reshape(source_image(:,((i-1)*5+j)),[32,32]);
    end
end
 figure (1);
 imshow(show_image/256);%需要归一化到0~1之间显示
%imwrite((show_image/256),'D:\digital image processing\KLtransform\result\ORL_32-32\Source_face.jpg');
[a,b]=size(source_image);
%% 由于Yale_B.mat的数据原本就以向量形式给出，因此不需要进行图像矩阵的向量化
%直接用mean函数求样本的平均脸
average_face=mean(source_image');
%imwrite(reshape(average_face,[32,32])/256,'D:\digital image processing\KLtransform\result\ORL_32-32\average_face.jpg');
figure (2);
imshow(reshape(average_face,[32,32])/256);
%% 求协方差矩阵
Cf=zeros(1024,1024);%初始化
for i=1:b
   Cf=Cf+(source_image(:,i)-average_face')*(source_image(:,i)-average_face')';
end
Cf=Cf/b;%得到30个样本的协方差矩阵
%tic
[V,D]=eig(Cf);%  returns diagonal matrix D of eigenvalues and matrix V whose columns are the corresponding right eigenvectors
%toc
%能够得到相应的特征值和按列排列的特征向量从小到大排列
B=V';
%对特征向量归一化
for i=1:1024
    A(i,:)=B(i,:)/sum(B(i,:).^2);
end
%可以得到离散K-L变换的式子为g=A(f-mf);
%% 对图像进行K-L变换的重建
%因为eig函数得到的特征值从大到小排列，由此得到的特征向量也按一定顺序排列
%先观测 D中特征值的大小 发现只有9个特征向量占比较大
for  k=1:length(K)
A1=A(((1024-K(k)+1):1024),:);%取相应的k的特征值的特征向量
%已知K-L变换的反变换为gk=Ak(f-mf);
for count=1:b %得到重构后的图像
gk=A1*(source_image(:,count)-average_face');
reconstruct_image(:,count)=A1'*gk+average_face';
end
%% 计算取k个特征值的均方误差
plus=0;
for i=(1025-K(k)):1024
    plus =plus+D(i,i);
end
ems(k)=sum(sum(D))-plus;
%save('ems_(k).mat','num2str(k)',ems);
for i=1:2
    for j=1:5
        show_reconstruct_image((1:32)+(j-1)*32,(1:32)+(i-1)*32)=reshape(reconstruct_image(:,((i-1)*5+j)),[32,32]);
    end
end
%imwrite((show_reconstruct_image/256),strcat('D:\digital image processing\KLtransform\result\ORL_32-32\',num2str(K(k)),'.jpg'));
end
plot(K,ems);%画误差曲线
axis([1 20 0 5e5]);
figure (3);
imshow(show_reconstruct_image/256);
end
%% 求特征值的快速算法 只局限于不同人脸的eigface的计算，对于同一张脸的不同姿势，很难看出不同

