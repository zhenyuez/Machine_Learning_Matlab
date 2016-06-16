function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

yy=zeros(size(y,1),size(Theta2,1));

for i=1:size(y,1)
    yy(i,y(i))=1;      
end

X1 = [ones(size(X,1),1) X];
A2 = sigmoid(X1*Theta1');
X2 = [ones(size(A2,1),1),A2];
A3 = sigmoid(X2*Theta2');

J = -1/m*(trace(yy'*log(A3))+trace((1.-yy')*log(1.-A3)));

%careful here since the theta for optimizing bias units are not included in
%cost fucntion.
for i=1:size(Theta1,1)
    for j=2:size(Theta1,2)
        J = J + Theta1(i,j)^2*lambda/(2*m);
    end    
end

for i=1:size(Theta2,1)
    for j=2:size(Theta2,2)
        J = J + Theta2(i,j)^2*lambda/(2*m);
    end    
end

% ----------back propagation------------------------------------------
del3=A3-yy; %might redefine it
del2=(del3*Theta2(:,2:size(Theta2,2))).*sigmoidGradient(X1*Theta1');

Del1=zeros(size(Theta1_grad));
Del2=zeros(size(Theta2_grad));

Del1=Del1+del2'*X1;
Del2=Del2+del3'*X2;

beta1=Theta1;
beta2=Theta2;

beta1=[zeros(size(beta1,1),1),beta1(:,2:size(beta1,2))];
beta2=[zeros(size(beta2,1),1),beta2(:,2:size(beta2,2))];
    
Theta1_grad=1/m*Del1+lambda/m*beta1;
Theta2_grad=1/m*Del2+lambda/m*beta2;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
