
function [Ktr, Kte] = featurecomb(Ktr, Kte, name,sn,NumberofTrainingData)


for loop=1:sn
    
    s2=sprintf('feature_%d.mat',loop);
    s2=[name,s2];
    load(s2);
H_train1=YYM_H(:,1:NumberofTrainingData);
H_test1=YYM_H(:,NumberofTrainingData+1:end);


Ktr = Ktr + eval_kernel(H_train1', H_train1', 'linear', 1);
Kte = Kte + eval_kernel(H_test1', H_train1', 'linear', 1);
clear YYM_H
end