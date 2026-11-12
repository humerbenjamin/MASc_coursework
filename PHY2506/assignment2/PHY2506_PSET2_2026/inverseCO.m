%----------------------------------------------------------
% Inverse modelling of MOPITT CO
%----------------------------------------------------------

%----------------------------------------------------------
% number of days for which we have MOPITT data
%----------------------------------------------------------
nday = 27;

%----------------------------------------------------------
% Dates for MOPITT data
%----------------------------------------------------------
dataday = [20010301, 20010302, 20010303, 20010304, 20010305, ...
           20010306, 20010307, 20010308, 20010309, 20010310, ...
           20010311, 20010312, 20010313, 20010314, 20010315, ...
           20010316, 20010317, 20010318, 20010319, 20010320, ...
           20010321, 20010322, 20010323, 20010324, 20010325, ...
           20010326, 20010327];
       
%----------------------------------------------------------
% size of state vector
%----------------------------------------------------------
nx = 7;

%----------------------------------------------------------
% Names of source regions
%----------------------------------------------------------
sources = char('China ','Korea','Japan','SEAsia','India','BoAsia','ROW');

Rdev = input('Enter the relative error for the MOPITT observations (default=0.2): ');
if isempty(Rdev)  ;
   Rdev=0.20; 
end
   
Bdev = input('Enter the relative error for the a priori emissions from the 7 source regions (default=0.5): ');
if isempty(Bdev)  ;
   Bdev=0.50; 
end
   
%----------------------------------------------------------
% a priori estimates of CO emissions (Tg CO/yr)
%----------------------------------------------------------
x_prior= double([125.1; 6.1; 7.7; 121.0; 84.2; 11.5; 2002.]);
x = x_prior;

%----------------------------------------------------------
% get diagonal a priori error covariance matrix
%----------------------------------------------------------
B = zeros(nx);
for n=1:nx;
    B(n,n) = (Bdev*x_prior(n))^2;
end
B_prior = B;

%----------------------------------------------------------
% begin looping through the data to sequentially update 
% the source estimates
%----------------------------------------------------------
prior_chisq = 0.;
totalobs = 0;
for it = 1:nday;
    filename = ['data/MOP_mod_',num2str(dataday(it)),'.data'];
    [tagco,y,ymod,maxobs] = readobs(filename);
    totalobs = totalobs + maxobs;
 
    innov = y - ymod;
    dx = x_prior - x;
    Binv = inv(B);
   
    R = zeros(maxobs);
    for n = 1:maxobs;
        R(n,n) = (Rdev*ymod(n))^2;
        prior_chisq = prior_chisq + ((innov(n)^2)/R(n,n));
    end
    
    H = zeros(maxobs,nx);
    for j = 1:maxobs;
        H(j,:) = tagco(j,:)./x_prior';
    end
    
    hess = ((H'*inv(R))*H) + inv(B);
    term = ((H'*inv(R))*innov) - (inv(B)*dx);
    incre= inv(hess)*term;
    x = x_prior + incre;
    B = inv(hess);
    
end

%----------------------------------------------------------
% ater sequential update, the a posteriori error covariance
% is stored in B
%----------------------------------------------------------
post_err = sqrt(diag(B));
prior_err = sqrt(diag(B_prior));

%----------------------------------------------------------
% get a posteriori correlation matrix
%----------------------------------------------------------
xcorr = zeros(nx,nx);
for i=1:nx;
  for j=1:nx;
     xcorr(i,j)= B(i,j)/sqrt(B(i,i)*B(j,j));
  end
end

results=table;
results.Sources = sources;
results.x = x;
results.x_prior = x_prior;
results.post_err = post_err;
results.prior_err = prior_err;

results

results_correlation=table;
results_correlation.Sources = sources;
results_correlation.China = num2str(xcorr(1:nx,1),2);
results_correlation.Korea = num2str(xcorr(1:nx,2),2);
results_correlation.Japan = num2str(xcorr(1:nx,3),2);
results_correlation.SEAsia = num2str(xcorr(1:nx,4),2);
results_correlation.India = num2str(xcorr(1:nx,5),2);
results_correlation.BoAsia = num2str(xcorr(1:nx,6),2);
results_correlation.ROW = num2str(xcorr(1:nx,7),2);

results_correlation

%----------------------------------------------------------
% calculate and plot averaging kernel matrix for the inversion
%----------------------------------------------------------
ave_kernel = eye(nx) - (B/B_prior);

ave_kernel

DOF=trace(ave_kernel);
DOF

plot([1:nx],ave_kernel');

ax=gca;
set(ax,'XTickLabel',{'China ','Korea','Japan','SEAsia','India','BoAsia','ROW'})

%----------------------------------------------------------
% output reduced chi-squared valu
%----------------------------------------------------------
reduced_chisq = prior_chisq/totalobs






%----------------------------------------------------------
% personal additions for calculating additional quantities
%----------------------------------------------------------

results_c = table;
results_c.Sources = sources;
results_c.x_prior = x_prior;
results_c.x = x;
results_c.increment = x - x_prior;
results_c.prior_error = prior_err;

results_c