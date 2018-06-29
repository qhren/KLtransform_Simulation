load('ORL_64x64.mat');
for i=1:120
    input=fea(1:120,:);
end
load('result.mat');
[a,b]=size(result);
for i=1:ceil(b/10)
    if(i<=10)
    possibility(i)=length(find(result((7:10)+(i-1)*10)==i))/4;
    else
    possibility(i)=length(find(isnan(result(101:end))==1))/(b-100);
    end
end
x=1:1:12;
bar(x,possibility);
xlabel('第几个人');
ylabel('识别率');
title('识别率直方图');
