%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)
% load data
load('ex4data1.mat');
m = size(X, 1);

fprintf('\nLoading Saved Neural Network Parameters ...\n')
% Load the weights into variables Theta1 and Theta2
load('ex4weights.mat');
% Unroll parameters 
nn_params = [Theta1(:) ; Theta2(:)];

fprintf('\nFeedforward Using Neural Network ...\n')
% Weight regularization parameter (we set this to 0 here).
lambda = 0;


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

fprintf(['Cost at parameters (loaded from ex4weights): %f '...
         '\n(this value should be about 0.287629)\n'], J);

         
lambda = 1;
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

J = J + r;

fprintf(['Cost at parameters (loaded from ex4weights): %f '...
         '\n(this value should be about 0.383770)\n'], J);

% sigmoid gradient
z = [-1 -0.5 0 0.5 1];
g_z = sigmoid(z);
temp = ones(size(g_z)) - g_z;
g = g_z .* temp;
fprintf('Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:\n  ');
fprintf('%f ', g);
fprintf('\n\n');