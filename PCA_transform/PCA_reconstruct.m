function []=PCA_reconstruct(M)
load('ORL_64x64.mat');% ��������
for i=1:10
    source_image(:,i)=fea(i,:);
end
%�������ʵ��������10*4����ʽ��ʾ
for i=1:5
    for j=1:2
        show_image((1:64)+(i-1)*64,(1:64)+(j-1)*64)=reshape(source_image(:,((i-1)*2+j)),[64,64]);
    end
end
figure (1);
imshow(show_image,[ ]);%��Ҫ��һ����0~1֮����ʾ
%imwrite((show_image/256),'D:\digital image processing\KLtransform\eigenfaces\Source_face.jpg');
[a,b]=size(source_image);
%% ����Yale_B.mat������ԭ������������ʽ��������˲���Ҫ����ͼ������������
%ֱ����mean������������ƽ����
average_face=mean(source_image');
% imwrite(reshape(average_face,[64,64])/256,'D:\digital image processing\KLtransform\eigenfaces\average_face.jpg');
figure (2);
imshow(reshape(average_face,[64,64]),[ ]);
% ��Э�������
Cf_temp=zeros(b,b);
Cf=zeros(4096,4096);%��ʼ��
for i=1:b
    source_temp=source_image(:,i)-average_face';
    Cf_temp=Cf_temp+(source_temp)'*source_temp;
    Cf=Cf+(source_temp)*source_temp';
end
Cf_temp=Cf_temp/b;
Cf=Cf/b;
%�õ��˼򻯵�Э�������
[V1,D1]=eig(Cf_temp);
%һ���˵�10��ͼƬ������9��������
sum1=zeros(4096,M);
%linear combinations
for i=1:M
    for m=1:b
    sum1(:,i)=sum1(:,i)+V1(m,i)*(source_image(:,m)-average_face');%�����ȥͼ��ľ�ֵ����Ҫ�����Ƚ����Ż�
%   source_image_new(:,m)=source_image(:,m)-average_face';
    end
end
for i=1:M
        eigenfaces((1:64),(1:64)+(i-1)*64)=reshape(sum1(:,i),[64,64]);
end
figure (3)
imshow(eigenfaces,[ ]);
%imwrite(eigenfaces/256,'D:\digital image processing\KLtransform\eigenfaces\eigen_faces.jpg');
%% ����projected onto face class��Ȩֵ
A1=sum1';
for count=1:b %��10�����������õ��ع����ͼ��
gk=A1*(source_image(:,count)-average_face');%KL�任�е�gk�൱��PCA�е�ӳ�䵽face class�е�Ȩ��
reconstruct_image(:,count)=A1'*gk+average_face';
end
% for i=1:10  %%�õ���Ӧ��Ȩֵ
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
