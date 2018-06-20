function [ ]=KLtransform(K)
%% ��ORL_32*32�е����ݽ���ͼ��ά�ؽ���ͬʱѡȡ��ͬ��Kֵ������ʵ����
% Background introduction
% ����ʵ������ݼ���ȡһ���˵�10��ͼƬ����ͼ��Ľ�ά�ؽ�
% ����arg kΪһ������
%% 
load('ORL_32x32.mat');% ��������
for i=1:10
    source_image(:,i)=fea(i,:);
end
%�������ʵ��������5*6����ʽ��ʾ
for i=1:2
    for j=1:5
        show_image((1:32)+(j-1)*32,(1:32)+(i-1)*32)=reshape(source_image(:,((i-1)*5+j)),[32,32]);
    end
end
 figure (1);
 imshow(show_image/256);%��Ҫ��һ����0~1֮����ʾ
%imwrite((show_image/256),'D:\digital image processing\KLtransform\result\ORL_32-32\Source_face.jpg');
[a,b]=size(source_image);
%% ����Yale_B.mat������ԭ������������ʽ��������˲���Ҫ����ͼ������������
%ֱ����mean������������ƽ����
average_face=mean(source_image');
%imwrite(reshape(average_face,[32,32])/256,'D:\digital image processing\KLtransform\result\ORL_32-32\average_face.jpg');
figure (2);
imshow(reshape(average_face,[32,32])/256);
%% ��Э�������
Cf=zeros(1024,1024);%��ʼ��
for i=1:b
   Cf=Cf+(source_image(:,i)-average_face')*(source_image(:,i)-average_face')';
end
Cf=Cf/b;%�õ�30��������Э�������
%tic
[V,D]=eig(Cf);%  returns diagonal matrix D of eigenvalues and matrix V whose columns are the corresponding right eigenvectors
%toc
%�ܹ��õ���Ӧ������ֵ�Ͱ������е�����������С��������
B=V';
%������������һ��
for i=1:1024
    A(i,:)=B(i,:)/sum(B(i,:).^2);
end
%���Եõ���ɢK-L�任��ʽ��Ϊg=A(f-mf);
%% ��ͼ�����K-L�任���ؽ�
%��Ϊeig�����õ�������ֵ�Ӵ�С���У��ɴ˵õ�����������Ҳ��һ��˳������
%�ȹ۲� D������ֵ�Ĵ�С ����ֻ��9����������ռ�Ƚϴ�
for  k=1:length(K)
A1=A(((1024-K(k)+1):1024),:);%ȡ��Ӧ��k������ֵ����������
%��֪K-L�任�ķ��任Ϊgk=Ak(f-mf);
for count=1:b %�õ��ع����ͼ��
gk=A1*(source_image(:,count)-average_face');
reconstruct_image(:,count)=A1'*gk+average_face';
end
%% ����ȡk������ֵ�ľ������
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
plot(K,ems);%���������
axis([1 20 0 5e5]);
figure (3);
imshow(show_reconstruct_image/256);
end
%% ������ֵ�Ŀ����㷨 ֻ�����ڲ�ͬ������eigface�ļ��㣬����ͬһ�����Ĳ�ͬ���ƣ����ѿ�����ͬ

