function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
% calculating cost
% calculating first part of cost expression (before "+" sign)
temp0 = X * theta;
temp1 = temp0 - y;
temp2 = power(temp1, 2);
temp3 = sum(temp2);
temp4 = temp3 / (2 * m);
% calculating second part of cost expression (after "+" sign)
theta(1) = 0;
temp5 = power(theta, 2);
temp6 = sum(temp5);
temp7 = (lambda * temp6) / (2 * m);
J = temp4 + temp7;


% calculating gradient
grad = ((1/m) * (X' * (temp1))) + ((lambda/m) * theta);
% =========================================================================

grad = grad(:);

end
