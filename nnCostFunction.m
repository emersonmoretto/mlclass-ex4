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

X = [ones(m,1) X];


% foward propagation
% a1 = X; 
a2 = sigmoid(Theta1 * X');
a2 = [ones(m,1) a2'];

h_theta = sigmoid(Theta2 * a2'); % h_theta equals z3

% y(k) - the great trick - we need to recode the labels as vectors containing only values 0 or 1 (page 5 of ex4.pdf)
yk = zeros(num_labels, m); 
for i=1:m,
  yk(y(i),i)=1;
end

% follow the form
J = (1/m) * sum ( sum (  (-yk) .* log(h_theta)  -  (1-yk) .* log(1-h_theta) ));



% Note that you should not be regularizing the terms that correspond to the bias. 
% For the matrices Theta1 and Theta2, this corresponds to the first column of each matrix.
t1 = Theta1(:,2:size(Theta1,2));
t2 = Theta2(:,2:size(Theta2,2));

% following the formula (theta ^ 2)
t1 = t1 .^ 2;
t2 = t2 .^ 2;

% regularization formula
Reg = lambda  * (sum( sum ( t1 )) + sum( sum ( t2 ))) / (2*m);

% cost function + reg
J = J + Reg;


% -------------------------------------------------------------

% Backprop

D1=0;
D2=0;

for t=1:m,

% dummie pass-by-pass

a1 = X(t,:); % X already have bias
%a1 = [1 a1]; %bias
z2 = Theta1 * a1';

a2 = sigmoid(z2);
a2 = [1 ; a2]; %bias

%a2 = [ones(m,1) a2'];
z3 = Theta2 * a2;

a3 = sigmoid(z3);

delta_3 = a3 - yk(y(t),t); % now the pig twist the tail


% god bless me
temp = (Theta2' * delta_3);
temp = temp(2:end);

delta_2 = temp .* sigmoidGradient(z2);

%delta_2 = delta_2 * sigmoidGradient(z2)';

%delta_2 = delta_2(2:end); % skipping sigma2(0) 

D2 = D2 + (delta_3 * a3');
D1 = D1 + (delta_2 * a2');


end;

Theta1_grad = D1/m;
Theta2_grad = D2/m;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
