function [TrainingTime,TrainingAccuracy,TestingAccuracy,label_index_expected,label_index_actual] = lastlayer(TrainingData_File, TestingData_File, Elm_Type, NumberofHiddenNeurons, ActivationFunction,kkk,C)
%FB-ELM是固定型单参数极限学习机
%是否用pinv(ones(1,kkk))根本就只是NB-ELM中当隐层节点为1时的精度呢？
%YYM中的数值其实只有一组而已 
% Usage: elm-MultiOutputRegression(TrainingData_File, TestingData_File, No_of_Output, NumberofHiddenNeurons, ActivationFunction)
% OR:    [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = elm-MultiOutputRegression(TrainingData_File, TestingData_File, No_of_Output, NumberofHiddenNeurons, ActivationFunction)
%
% Input:
% TrainingData_File     - Filename of training data set
% TestingData_File      - Filename of testing data set
% No_of_Output          - Number of outputs for regression
% NumberofHiddenNeurons - Number of hidden neurons assigned to the ELM
% ActivationFunction    - Type of activation function:
%                           'sig' for Sigmoidal function
%                           'sin' for Sine function
%                           'hardlim' for Hardlim function
%kkk                   -number of hidden nodes
%
% Output: 
% TrainingTime          - Time (seconds) spent on training ELM
% TestingTime           - Time (seconds) spent on predicting ALL testing data
% TrainingAccuracy      - Training accuracy: 
%                           RMSE for regression
% TestingAccuracy       - Testing accuracy: 
%                           RMSE for regression

%
    %%%%    Authors:    MR QIN-YU ZHU AND DR GUANG-BIN HUANG
    %%%%    NANYANG TECHNOLOGICAL UNIVERSITY, SINGAPORE
    %%%%    EMAIL:      EGBHUANG@NTU.EDU.SG; GBHUANG@IEEE.ORG
    %%%%    WEBSITE:    http://www.ntu.edu.sg/eee/icis/cv/egbhuang.htm
    %%%%    DATE:       APRIL 2004

%%%%%%%%%%% Load training dataset
%train_data=load(TrainingData_File);
%train_data=train_data(:,2:9);

%%%%%%%%%%% Macro definition
REGRESSION=0;
CLASSIFIER=1;


%%%%%%%%%%% Load training dataset
train_data=TrainingData_File;
T=train_data(:,1)';
aaa=T;
P=train_data(:,2:size(train_data,2))';
clear train_data;                                   %   Release raw training data array

%%%%%%%%%%% Load testing dataset
test_data=TestingData_File;
TV.T=test_data(:,1)';
TV.P=test_data(:,2:size(test_data,2))';
clear test_data;                                    %   Release raw testing data array

NumberofTrainingData=size(P,2);
NumberofTestingData=size(TV.P,2);
NumberofInputNeurons=size(P,1);

if Elm_Type~=REGRESSION
    %%%%%%%%%%%% Preprocessing the data of classification
    sorted_target=sort(cat(2,T,TV.T),2);
    label=zeros(1,1);                               %   Find and save in 'label' class label from training and testing data sets
    label(1,1)=sorted_target(1,1);
    j=1;
    for i = 2:(NumberofTrainingData+NumberofTestingData)
        if sorted_target(1,i) ~= label(1,j)
            j=j+1;
            label(1,j) = sorted_target(1,i);
        end
    end
    number_class=j;
    NumberofOutputNeurons=number_class;
    
    %%%%%%%%%% Processing the targets of training
    temp_T=zeros(NumberofOutputNeurons, NumberofTrainingData);
    for i = 1:NumberofTrainingData
        for j = 1:number_class
            if label(1,j) == T(1,i)
                break; 
            end
        end
        temp_T(j,i)=1;
    end
    T=temp_T*2-1;

    %%%%%%%%%% Processing the targets of testing
    temp_TV_T=zeros(NumberofOutputNeurons, NumberofTestingData);
    for i = 1:NumberofTestingData
        for j = 1:number_class
            if label(1,j) == TV.T(1,i)
                break; 
            end
        end
        temp_TV_T(j,i)=1;
    end
    TV.T=temp_TV_T*2-1;

end                                                 %   end if of Elm_Type



D_YYM=[];
D_Input=[];
D_beta=[];
D_beta1=[];
TY=[];
FY=[];
BiasofHiddenNeurons1=[];
Y=zeros*T;
%{
clear  TestingData_File
clear   TrainingData_File
save matlab TV -v7.3
clear TV
%}
E1=T;
for i=1:kkk











Y2=E1;



switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        [Y22,PS(i)]=mapminmax(Y2,0.01,0.99);
    case {'sin','sine'}
        %%%%%%%% Sine
       [Y22,PS(i)]=mapminmax(Y2,0,1);
end


Y2=Y22';

start_time_train=cputime;
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
Y4=(-log((1./Y2)-1))'; 


    case {'sin','sine'}
        %%%%%%%% Sine
       Y3=asin(Y2)';
end





%YYM=inv(eye(size(P,1))/C+P * P') * P * Y4';

YYM=(eye(size(P,1))/C+P * P') \ P *Y4';

%YYM=pinv(P)' *Y4';
YJX=P'*YYM;

BB1=size(Y4);
BB2=sum(YJX-Y4');
BB(i)=BB2(1)/BB1(2);
end_time_train=cputime;
TrainingTime=end_time_train-start_time_train;

GXZ111=P'*YYM-BB(i);
%{
cc=(eye(size(P,1))/C+P * P') \ P* (GXZ1-Y4');



GXZ11=P'*(YYM-cc)-BB(i);
GXZ111=GXZ11;
%}
%}
%{
BBB(i)=mean(GXZ11-Y4');
GXZ111=P'*(YYM-cc)-BB(i)-BBB(i);
BBBB(i)=BB(i)+BBB(i);

%}
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
GXZ2=1./(1+exp(-GXZ111'));
    case {'sin','sine'}
        %%%%%%%% Sine
GXZ2=sin(GXZ111');
end


FYY = mapminmax('reverse',GXZ2,PS(i));



FT1=FYY;%*ones(1,kkk)';
TrainingAccuracy=sqrt(mse(FT1-E1));
E1=E1-FT1;

Y=Y+FYY;

D_YYM{i}=YYM;

if Elm_Type == CLASSIFIER
        MissClassificationRate_Training=0;
    for i = 1 : size(T, 2)
        [x, label_index_expected]=max(T(:,i));
        [x, label_index_actual]=max(Y(:,i));
        if label_index_actual~=label_index_expected
            MissClassificationRate_Training=MissClassificationRate_Training+1;
        end
    end
    TrainingAccuracy=1-MissClassificationRate_Training/size(T,2);
end

end

%load matlab

start_time_test=cputime;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
TY2=zeros*TV.T;

for i=1:kkk
GXZ1=D_YYM{i}'*TV.P-BB(i);
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
GXZ2=1./(1+exp(-GXZ1'));
    case {'sin','sine'}
        %%%%%%%% Sine
GXZ2=sin(GXZ1');
end

FYY = mapminmax('reverse',GXZ2',PS(i));
%FYY=GXZ2;


TY2=TY2+FYY;
E1=TY2-TV.T;
TestingAccuracy=sqrt(mse(E1));
end


end_time_test=cputime;
 test_time=end_time_test-start_time_test;


if Elm_Type == CLASSIFIER
%%%%%%%%%% Calculate training & testing classification accuracy

    MissClassificationRate_Testing=0;


    for i = 1 : size(TV.T, 2)
        [x, label_index_expected(i)]=max(TV.T(:,i));
        [x, label_index_actual(i)]=max(TY2(:,i));
        if label_index_actual(i)~=label_index_expected(i)
            MissClassificationRate_Testing=MissClassificationRate_Testing+1;
        end
    end
    TestingAccuracy=1-MissClassificationRate_Testing/size(TV.T,2)  ;


end
