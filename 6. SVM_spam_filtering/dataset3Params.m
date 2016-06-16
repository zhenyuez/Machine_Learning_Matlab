function [C, sigma] = dataset3Params(X, y, Xval, yval)
%dataset3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = dataset3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
%C = 1;
%sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

cv = [0.01,0.03,0.1,0.3,1,3,10,30];
sv = [0.01,0.03,0.1,0.3,1,3,10,30];
err=zeros(8,8);

for i=1:8
    for j=1:8
    model = svmTrain(X, y, cv(i), @(x1, x2) gaussianKernel(x1, x2, sv(j))); 
    pred = svmPredict(model, Xval);
    err(i,j) = mean(double(pred~=yval));

%err(i,j)=0;
%    for k=1:size(pred,1)
%        if pred(k)~=yval(k)
%        err(i,j) = err(i,j)+1;
%        end
%    end
%    err(i,j)=err(i,j)/size(pred,1);
    
    end
end

[~, I] = min(err(:));
[I_row, I_col] = ind2sub(size(err),I);

C=cv(I_row);
sigma=sv(I_col);

% =========================================================================

end
