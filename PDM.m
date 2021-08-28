function [Fm,Gm]=PDM(train,test,C,K)
%%%%%%%%%%%%%%%%%
%Input:
%train: s*n, where s indicates the number of training instances, n indicates the number of features, and the last column indicates the class label feature
%test:  q*n, where q indicates the number of testing  instances, n indicates the number of features, and the last column indicates the class label feature
%C: the number of classes
%K: a parameter to indicate the number of neighbors,e.g.,1 indicates sqrt(s),2 indicates 2*sqrt(s)
%Output:
%Fm: F-measure in testing instances
%Gm: G-mean in testing instances
%Copyright: 
%Prof.Hualong Yu, Jiangsu University of Science and Technology, China
%%%%%%%%%%%%%%%%%Training procedure
[s,n]=size(train);
%Dividing data subsets for each class
NC=zeros(1,C); % reserving the number of instances in each class
train_subsets=[]; % reseving the training subsets divided based on class labels
for i=1:s
  for j=1:C
    if(train(i,end)==j)
       NC(1,j)=NC(1,j)+1;
       train_subsets(NC(1,j),:,j)=train(i,:);
    end
  end
end
[s1,s2]=sort(NC); %ranking to find the minimal class
ad_factor=[];  % reserving the adjust factor in each class for calculating the nomalization probability density
lamda=[]; % reserving the neighborhood parameter for each class
sigma=[]; % reseving the normalization factor for each class
for i=1:C
   ad_factor(1,i)=NC(1,i)/s1(1,1);
   lamda(1,i)=ceil(K*sqrt(NC(1,i)));
   sigma(1,i)=KNN_alike(train_subsets(:,1:(end-1),i),NC(1,i),lamda(1,i)); % call KNN-PDE-alike algorithm to calculate normalization factor for the corresponding class
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%Testing Procedure
[q,n]=size(test);
posterior_test=[]; % reserving the relative probability density for each testing instance in each class
for i=1:q
   for j=1:C
        dis=[];
        for k=1:NC(1,j)
            dis(1,k)=norm(test(i,1:(end-1))-train_subsets(k,1:(end-1),j),2);% calculating Euclidean distance beteen ith testing instance and kth instance in jth class
        end
        [s1,s2]=sort(dis);
        posterior_test(i,j)=((1/s1(1,lamda(1,j)))/sigma(1,j))*ad_factor(1,j);
   end
end

%%%%%%%%%%%Evaluating the results
F_measure=0;
G_mean=1;
for i=1:C
   TP=0;
   TN=0;
   FN=0;
   FP=0;
   for j=1:q
      [s1,s2]=dsort(posterior_test(j,:));
      if(test(j,end)==i && s2(1,1)==i)
         TP=TP+1;
      elseif (test(j,end)==i && s2(1,1)~=i)
         FN=FN+1;
      elseif (test(j,end)~=i && s2(1,1)==i)
         FP=FP+1;
      else
         TN=TN+1;
      end
   end
   if(TP==0 && FN==0)
       recall=1;
   else
       recall=TP/(TP+FN);
   end
   if(TP==0 && FP==0)
       precision=0;
   else
       precision=TP/(TP+FP);
   end
   if(recall==0 && precision==0)
       F_measure=F_measure+0;
   else
       F_measure=F_measure+(2*precision*recall)/(precision+recall);
   end
   G_mean=G_mean*recall;
end
Fm=F_measure/C;
Gm=nthroot(G_mean,C);