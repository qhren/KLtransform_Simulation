function []=PCA_reconstruct(M)
load('ORL_64x64.mat');% 载入数据
for i=1:10
    source_image(:,i)=fea(i,:);
end
%将总体的实验样本以10*4的形式显示
for i=1:5
    for j=1:2
        show_image((1:64)+(i-1)*64,(1:64)+(j-1)*64)=reshape(source_image(:,((i-1)*2+j)),[64,64]);
    end
end
figure (1);
imshow(show_image,[ ]);%需要归一化到0~1之间显示
%imwrite((show_image/256),'D:\digital image processing\KLtransform\eigenfaces\Source_face.jpg');
[a,b]=size(source_image);
%% 由于Yale_B.mat的数据原本就以向量形式给出，因此不需要进行图像矩阵的向量化
%直接用mean函数求样本的平均脸
average_face=mean(source_image');
% imwrite(reshape(average_face,[64,64])/256,'D:\digital image processing\KLtransform\eigenfaces\average_face.jpg');
figure (2);
imshow(reshape(average_face,[64,64]),[ ]);
% 求协方差矩阵
Cf_temp=zeros(b,b);
Cf=zeros(4096,4096);%初始化
for i=1:b
    source_temp=source_image(:,i)-average_face';
    Cf_temp=Cf_temp+(source_temp)'*source_temp;
    Cf=Cf+(source_temp)*source_temp';
end
Cf_temp=Cf_temp/b;
Cf=Cf/b;
%得到了简化的协方差矩阵；
[V1,D1]=eig(Cf_temp);
%一个人的10张图片最多具有9张特征脸
sum1=zeros(4096,M);
%linear combinations
for i=1:M
    for m=1:b
    sum1(:,i)=sum1(:,i)+V1(m,i)*(source_image(:,m)-average_face');%如果减去图像的均值，需要对亮度进行优化
%   source_image_new(:,m)=source_image(:,m)-average_face';
    end
end
for i=1:M
        eigenfaces((1:64),(1:64)+(i-1)*64)=reshape(sum1(:,i),[64,64]);
end
figure (3)
imshow(eigenfaces,[ ]);
%imwrite(eigenfaces/256,'D:\digital image processing\KLtransform\eigenfaces\eigen_faces.jpg');
%% 计算projected onto face class的权值
A1=sum1';
for count=1:b %用10个特征向量得到重构后的图像
gk=A1*(source_image(:,count)-average_face');%KL变换中的gk相当于PCA中的映射到face class中的权重
reconstruct_image(:,count)=A1'*gk+average_face';
end
% for i=1:10  %%得到相应的权值
%     for k=1:M
%         omiga(i,k)=(sum1(:,k))'*(source_image(:,i)-average_face');
%     end
% end
% for count=1:b
%     reconstruct_image(:,count)=A1'*omiga(count,:)'+average_face';
% end
for i=1:5
    for j=1:2
        show_reconstruct_image((1:64)+(i-1)*64,(1:64)+(j-1)*64)=reshape(reconstruct_image(:,((i-1)*2+j)),[64,64]);
    end
end
% figure (4);
% imshow(show_reconstruct_image,[]);
% imwrite(show_reconstruct_image/256,'D:\digital image processing\KLtransform\eigenfaces\reconstruct_image.jpg');
end
