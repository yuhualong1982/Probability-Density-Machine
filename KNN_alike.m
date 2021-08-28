function sigma=KNN_alike(data,n_instances,K)
%%%%%%%%%%%%
%Input:
%data: m*n, where m denotes the number of instances, and n denotes the number of attributes
%n_instances: indicating how many instances should be calculated
%K:neighborhood parameter to calculate the relative probability density
%Output:
%sigma: the normalization factor for data
%Copyright: 
%Prof.Hualong Yu, Jiangsu University of Science and Technology, China
%%%%%%%%%%%%%%%%%
% calculating Kth neighbor distance for each instance
Kiwi=[]; % reserving the reciprocal of Kth distance for each instance
for i=1:n_instances
  dis=[];
  for j=1:n_instances
     dis(1,j)=norm(data(i,:)-data(j,:),2);% calculating Euclidean distance beteen ith and jth instance
  end
  [s1,s2]=sort(dis);
  Kiwi(1,i)=1/s1(K+1);
end
sigma=sum(Kiwi); %calculating the normalization factor
  