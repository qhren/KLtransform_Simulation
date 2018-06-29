function [  ]=face_rec(input)
%% step1 & step2 &step3
%mkdir('D:\digital image processing\KLtransform\result\','face_recognition');
load('ORL_64x64.mat');
%input=fea(1:120,:);
[row,col]=size(input);
M=10;%ѡȡ10����
b=60;%ͼ������Ϊ60��
for i=1:M
    for j=1:6
    source_image(:,((i-1)*6+j))=fea(((i-1)*10+j),:);
    end
end
%�������ʵ��������4*10����ʽ��ʾ
for i=1:M
    for j=1:6
        show_image((1:64)+(j-1)*64,(1:64)+(i-1)*64)=reshape(source_image(:,((i-1)*6+j)),[64,64]);
    end
end
% figure (1);
% imshow(show_image/256);%��Ҫ��һ����0~1֮����ʾ
%imwrite((show_image/256),'D:\digital image processing\KLtransform\result\face_recognition\Source_face.jpg');
[a,b]=size(source_image);
%% ����Yale_B.mat������ԭ������������ʽ��������˲���Ҫ����ͼ������������
%ֱ����mean������������ƽ����
average_face=mean(source_image');
% imwrite(reshape(average_face,[64,64])/256,'D:\digital image processing\KLtransform\result\face_recognition\average_face.jpg');
% figure (2);
% imshow(reshape(average_face,[64,64])/256);
% ��Э�������
%��ʼ��
Cf_temp=zeros(b,b);
for i=1:b
    source_temp=source_image(:,i)-average_face';
    Cf_temp=Cf_temp+(source_temp)'*source_temp;
%    Cf=Cf+(source_temp)*source_temp';
end
Cf_temp=Cf_temp/b;
% Cf=Cf/b;
%�õ��˼򻯵�Э�������
[V1,D1]=eig(Cf_temp);
sum1=zeros(4096,10);
V=zeros(4096,b);
%linear combinations �õ�Cf����������,��Ϊ��10���ˣ�����ȡ������������ĿΪ10
for i=1:M
    for m=1:b  %%ȡ10����40��
    sum1(:,i)=sum1(:,i)+V1(m,i)*(source_image(:,m)-average_face');%�����ȥͼ��ľ�ֵ����Ҫ�����Ƚ����Ż�
%   source_image_new(:,m)=source_image(:,m)-average_face';
    end
end
for i=1:b
    for m=1:b
    V(:,i)=V(:,i)+V1(m,i)*(source_image(:,m)-average_face');%�����ȥͼ��ľ�ֵ����Ҫ�����Ƚ����Ż�
%   source_image_new(:,m)=source_image(:,m)-average_face';
    end
end
for i=1:5
    for j=1:2
       faces_rec((1:64)+(i-1)*64,(1:64)+(j-1)*64)=reshape(sum1(:,((i-1)*2+j)),[64,64]);
    end
end
for i=1:M
    for j=1:6
       eigen_faces((1:64)+(j-1)*64,(1:64)+(i-1)*64)=reshape(V(:,((i-1)*6+j)),[64,64]);
    end
end
% д��
% imwrite((faces_rec/256),'D:\digital image processing\KLtransform\result\face_recognition\faces_rec.jpg');
% imwrite((eigen_faces/256),'D:\digital image processing\KLtransform\result\face_recognition\eigen_faces.jpg');
% figure (3); imshow(eigen_faces/256);
% figure (4); imshow(faces_rec/256);
%%  step4  how to calculate the fourth step remains to be done.
%calculate omiga
for i=1:b
    for k=1:M
        omiga(i,k)=(sum1(:,k))'*(source_image(:,i)-average_face');
    end
end
for i=1:M
    average_omiga(i,:)=mean(omiga((1+6*(i-1)):6*i,:),1);
end
for j=1:b
threshold(j,:)=norm(omiga(j,:)-average_omiga(ceil(j/6),:));
end
for i=1:M
threshold_(i)=max(threshold(((1:6)+6*(i-1)),:));
end
%% ����ŷ�Ͼ���
temp3=zeros(4096,b);
for i=1:b
    for j=1:M
        temp3(:,i)=omiga(i,j)*sum1(:,j)+temp3(:,i);
    end
end
for i=1:b
    threshold1_(i)=norm(source_image(:,i)-average_face'-temp3(:,i));
end
limit=max(threshold1_);
%% ��������ͼ����ж�
% Ȩֵ����
for count=1:row
    count
    f=input(count,:)';
 for k=1:M
        omiga_(k)=(sum1(:,k))'*(f-average_face');
 end
% ŷ�Ͼ�����㣨face_space,face_class)
temp4=zeros(4096,1);
 for j=1:M
        temp4=omiga_(j)*sum1(:,j)+temp4;
end
distance_faceSpace=norm(f-average_face'-temp4);
if(distance_faceSpace<=limit*1.1) %����ֵ���ϲ���10%
    for i=1:M
      temp5(i)=norm((omiga_-average_omiga(i,:)),2);
    end
      K=find(temp5==min(sort(temp5)));
      if(temp5(K)<=threshold_(K)*1.1) %����ֵ���ϲ���10%
        fprintf('��ͼ�����ڵ�%d����\n',K);
        result(count)=K;
      else
        result(count)=nan;
        disp('unknown face');
      end
else
   disp('This is not a face image');
   result(count)=0;
end
end
save('result.mat','result');
end

