function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

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
J = -s / m;

% calculate gradient
for j = 1:length(grad)
	s = 0;
	for i = 1:size(h)
		s = s + ((h(i) - y(i)) * X(i, j));
	endfor;
	grad(j) = s / m;
endfor;
% =============================================================

end
