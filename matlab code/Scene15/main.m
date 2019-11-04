

%%%%%%%%%%% Running this code needs 16 GB memory, and 1 to 2 minutes %%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Ktr=0;Kte=0;  % Ktr denotes training features extracted from hundreds of subnetwork nodes. 
%%%%%%%%%%%%%%%  Kte denotes testing features extracted from hundreds of subnetwork nodes.


 %%%%%%%%%%%%%%%%%%%%%%%  Spatial pyramid features loading for Scene15 dataset  %%%%%%%%%%%%%%%%%%
load spatialpyramidfeatures4scene15.mat
[labelMat,b]=find(labelMat==1);
fae=[labelMat'; featureMat]';

clear featureMat
train_per_image=100;   %% training image per class
 
    sample_G;
    clear fae



Training=[tr_fea];
Testing=[ts_fea];

for loop=1:10  %% loop denotes data-channel number
    
name=sprintf('scene15_channel_%d',loop);
num_subnetwork_node=3;  %%%%% 3 subnetework nodes used %%%%%%%%%%%%
dimension=100;   %%%%%%%%%%%% dimensionality (100) used in each subnetwork node %%%%%%%%%%%%%
C1=2^8;      %%%%%% parameter C in equation (5) %%%%%%%%%%%

[train_time,NumberofTrainingData]=Layerfirst(Training,Testing,1,dimension,'sine',C1,2,num_subnetwork_node,name); %%%%% subspace features $i$ will be saved as 'scene15_channel_i_feature_*.mat'


%%%%%%%%%%%%%%%%%%%%%%%% Layer 2: kernel combination %%%%%%%%%%%%%%%%%%%%
[Ktr, Kte] = featurecomb(Ktr, Kte,name,3,NumberofTrainingData);
end




Training=[Training(:,1) Ktr];   %%%%%% here Ktr denotes final features for training 
 Testing=[Testing(:,1) Kte];   %%%%%% Kte denotes final feature for testing 
 
%%%%%%%%%%%%%%%%%%%%%%%%%% Layer 3: classifier with subnetwork nodes %%%%%%
 C2=2^12;    %%% Parameter C in equation (15)
[train_time,  train_accuracy11,test_accuarcy]=lastlayer(Training,Testing,1,1,'sig',1,C2);   

test_accuarcy  %%%%  %%%%  If we set dimension=20000, C1=2^2, loop=1:1, the test_accuracy is about 98.7% (Training time 1168 second)





 







