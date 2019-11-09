function [NumberofTrainingData] = Layerfirst_C256(train_data, test_data, Elm_Type, NumberofHiddenNeurons, ActivationFunction,C,kkkk,sn,Categories_number,name,filter,index)


%%%%%%%%%%% Macro definition
REGRESSION=0;
CLASSIFIER=1;
fdafe=0;
%%%%%%%%%%% Load training dataset

T=train_data(:,1:Categories_number)';
P=train_data(:,Categories_number+1:end)';
OrgP=P;
clear train_data;                                   %   Release training data array

%%%%%%%%%%% Load testing dataset

TV.T=test_data(:,1:Categories_number)';
TV.P=test_data(:,Categories_number+1:end)';
clear test_data;                                    %   Release testing data array

NumberofTrainingData=size(P,2);
NumberofTestingData=size(TV.P,2);
NumberofInputNeurons=size(P,1);

%%%%%%%%%%% Calculate weights & biases


for subnetwork=1:sn

    
    
for j=1:kkkk
    if j==4
        fdafe=1;
    end
    if j==1
        count=1;
    else
        count=1;
    end
    
%%%%%%%%%%% Random generate input weights InputWeight (w_i) and biases BiasofHiddenNeurons (b_i) of hidden neurons
for nxh=1:count
    
if j==1
    
    BiasofHiddenNeurons1=rand(NumberofHiddenNeurons,1);
       BiasofHiddenNeurons1=orth(BiasofHiddenNeurons1);
InputWeight1=rand(NumberofHiddenNeurons,NumberofInputNeurons)*2-1;

        if NumberofHiddenNeurons > NumberofInputNeurons
            InputWeight1 = orth(InputWeight1);
        else
            InputWeight1 = orth(InputWeight1')';
        end


tempH=InputWeight1*P;
ind=ones(1,NumberofTrainingData);
BiasMatrix=BiasofHiddenNeurons1(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH=tempH+BiasMatrix;
else


        
    switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
PP1=(-log((1./PP)-1)); 


    case {'sin','sine'}
        %%%%%%%% Sine
    %   PP1=asin(PP);
     PP1=atan(PP);
   % PP1=atanh(PP);
    end
    
clear PP

    PP1=real(PP1);
    
    P=OrgP;
    clear H
   InputWeight_temp=((eye(size(P,1))/C(1)+P * P') \ P *PP1')';

      [r,c]=size(InputWeight_temp);
totalNum=r*c;
randomindex=1+floor(rand(1,floor(totalNum*0.5))*totalNum);
InputWeight_temp(randomindex)=0; 
   
       InputWeight1= InputWeight1+0.5*InputWeight_temp';   %Learning rate=0.5
   
   fdafe=0;
   tempH=InputWeight1*P;
   YYM_H=InputWeight1*[P TV.P];
BB1=size(PP1);
BB2=sum(sum(tempH-PP1));
   clear PP1
BBP=BB2/BB1(2);
tempH= bsxfun(@minus, tempH', BBP.')';
YYM_tempH= bsxfun(@minus, YYM_H', BBP.')';
clear YYM_H
end



%tempH=tempH+BiasMatrix;

%%%%%%%%%%% Calculate hidden neuron output matrix H

        %%%%%%%% Sine
         H = sin(tempH);       
          

clear tempH;                                        %   Release the temparary array for calculation of hidden neuron output matrix H
clear  BiasMatrix 


   

if j>1
    
     YYM_H = sin(YYM_tempH);  

%YYM_H=mapminmax(YYM_H,-1,1);
H=YYM_H(:,1:NumberofTrainingData);
feature_total=YYM_H;
       
   if isempty(index)==1


s1=sprintf('filter_%d.mat',filter);
s1=[name,s1];
s2=sprintf('subnetwork_%d.mat',subnetwork);
s2=[s1,s2];
%save(s2,'feature_total','-mat');
save(s2,'feature_total','-mat','-v7.3');
clear feature_total

       
   else
 

    if index.filters{filter}.subnetworks(subnetwork)==1 

s1=sprintf('filter_%d.mat',filter);
s1=[name,s1];
s2=sprintf('subnetwork_%d.mat',subnetwork);
s2=[s1,s2];
%save(s2,'feature_total','-mat');
save(s2,'feature_total','-mat','-v7.3');
clear feature_total
    end

   end
   
   



end


end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



P=H;
clear H
E1=T;

for i=1:2


Y2=E1;

clear tempH


if fdafe==0
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        [Y22,PS(1)]=mapminmax(Y2,0.01,0.99);
    case {'sin','sine'}
        %%%%%%%% Sine
       [Y22,PS(1)]=mapminmax(Y2,-1,1);
end
else

Y22 = mapminmax.apply(Y2,PS(1));


end


Y2=Y22;

switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
Y4=(-log((1./Y2)-1)); 


    case {'sin','sine'}
        %%%%%%%% Sine
       Y4=asin(Y2);
end

Y4=real(Y4);
clear Y2
clear Y22

if fdafe==0

YYM=(eye(size(P,1))/C(2)+P * P') \ P *Y4';


YJX=(YYM'*P)';
else
    
PP=(Y4'*((eye(size(YYM,1))/C(2)+YYM * YYM') \ YYM )')';
YJX=PP'*YYM;

end




BB1=size(Y4);
BB2=sum(YJX-Y4');
BB=BB2/BB1(2);
BB=BB(1);

clear Y4




GXZ111 = bsxfun(@minus, YJX', BB.')';


switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
GXZ2=1./(1+exp(-GXZ111'));
    case {'sin','sine'}
        %%%%%%%% Sine
GXZ2=sin(GXZ111');
end

clear YJX



FYY = mapminmax('reverse',GXZ2,PS(1))';
clear GXZ2





E1=E1-FYY';

if i==1
    T_L=E1;

end


if i==1
fdafe=1;
end


end
PP=P+PP;
PP=mapminmax(PP,-1,1);
end
T=E1;
P=OrgP;
fdafe=0;
end
clear
