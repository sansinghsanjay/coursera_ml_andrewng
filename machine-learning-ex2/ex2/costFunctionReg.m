function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
% transpose theta
theta = theta';

% calculate z
z = zeros(size(X)(1), 1);
k = 1;
for i = 1:size(X)(1)
	for j = 1:size(X)(2)
		z(k, 1) = z(k, 1) + theta(1, j) * X(i, j);
	endfor;
	k = k + 1;
endfor;

% calculate h(x)
h = sigmoid(z);

% calculate cost
s = 0;
for i = 1:size(h)
	s = s + (y(i) * log(h(i))) + ((1 - y(i)) * log(1 - h(i)));
endfor;
theta_sum = 0
for j = 2:length(theta)
  theta_sum = theta_sum + power(theta(1, j), 2);
endfor;
J = (-s / m) + ((lambda / (2 * m)) * theta_sum);

% calculate gradient
for j = 1:length(grad)
	s = 0;
	for i = 1:size(h)
    if(j == 1)
		  s = s + ((h(i) - y(i)) * X(i, j));
    else
      s = s + ((h(i) - y(i)) * X(i, j)) + ((lambda / m) * theta(1, j));
    endif;
	endfor;
	grad(j) = s / m;
endfor;
% =============================================================

end
