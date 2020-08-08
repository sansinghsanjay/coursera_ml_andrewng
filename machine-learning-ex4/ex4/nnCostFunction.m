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
% Calculating cost without regularization
X = [ones(m, 1) X];
for i=1:m
  y_onehot = zeros(num_labels, 1);
  y_onehot(y(i)) = 1;
  % first layer of neural network
  a2 = X(i, :) .* Theta1;
  a2 = sum(a2');
  z2 = sigmoid(a2);
  z2 = [ones(size(z2, 1), 1) z2];
  % second layer of neural network
  a3 = z2 .* Theta2;
  a3 = sum(a3');
  z3 = sigmoid(a3);
  for j=1:num_labels
    J = J + (y_onehot(j) * log(z3(j)) + (1 - y_onehot(j)) * log(1 - z3(j)));
  endfor;
endfor;
J = -J / m;

% Calculating regularization
r = 0;
for j=1:size(Theta1, 1)
  for k=2:size(Theta1, 2)
    r = r + power(Theta1(j, k), 2);
  endfor;
endfor;
for j=1:size(Theta2, 1)
  for k=2:size(Theta2, 2)
    r = r + power(Theta2(j, k), 2);
  endfor;
endfor;
r = (lambda / (2 * m)) * r;

% adding cost and regularization
J = J + r;

% implementing Backpropagation
for i = 1:m
    %The raw neuros counts the bias unit, which means it as a0 -> a25, not count a1
    a1 = X(i,:)'; % insert the bias unit
    z2 = Theta1 * a1;
    a2 = [1 ; sigmoid(z2)];
    z3 = Theta2 * a2;
    a3 = sigmoid(z3);
    hx = a3;
    %y_vec = zeros(num_labels,1);
    %y_vec(y(i)) = 1; % this the compare y to be a binary vector
    %one_query_cost = (-y_vec' * log(hx) - (1 - y_vec') * log(1 - hx));
    %J = J + one_query_cost;
    y_vec = zeros(num_labels,1);
    y_vec(y(i)) = 1; % this the compare y to be a binary vector
    delta3 = (a3 - y_vec);
    
    g_z2 = sigmoid(z2);
    temp = ones(size(g_z2)) - g_z2;
    g_dash_z2 = g_z2 .* temp;
  
    delta2 = (Theta2(:,2:end)' * delta3) .* g_dash_z2;
    Theta2_grad = Theta2_grad + delta3 * a2';
    Theta1_grad = Theta1_grad + delta2 * a1';
endfor;
%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad / m;
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
