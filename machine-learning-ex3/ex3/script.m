%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

load('ex3data1.mat');
m = size(X, 1);

fprintf('\nLoading Saved Neural Network Parameters ...\n')

% Load the weights into variables Theta1 and Theta2
load('ex3weights.mat');


num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);


% Add ones to the X data matrix
X = [ones(m, 1) X];
A1 = zeros(size(X, 1), size(Theta1, 1));
A2 = zeros(size(X, 1), size(Theta2, 1));
for i = 1:m
  temp1 = X(i, :) .* Theta1;
  temp2 = sum(temp1');
  temp3 = sigmoid(temp2);
  A1(i, :) = temp3;
endfor;
A1 = [ones(size(A1, 1), 1) A1];
for i = 1:m
  temp1 = A1(i, :) .* Theta2;
  temp2 = sum(temp1');
  temp3 = sigmoid(temp2);
  A2(i, :) = temp3;
endfor;
[max_values pred] = max(A2');
pred = pred';
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);