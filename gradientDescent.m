function [theta] = gradientDescents(X, y, theta, alpha, num_iters)

% The number of training data.
m = length(y)
for iter = 1 : num_iters
    theta_temp = theta;
    for j = 1 : length(theta)
        % Variety 'temp' means sum of subtrahend of equation (13)
        temp = 0;
        for i = 1 : length(X)
            % 'L_function' represents the logistic function.
            % Implementation of equation (13)
            L_function = 1 / (1 + exp(-dot(theta', X(i, :))));
            temp = temp + alpha / length(theta) * dot((L_function - y(i)), X(i, j));
        end
        theta_temp(j) = theta(j) - temp;
    end
    theta = theta_temp;
end