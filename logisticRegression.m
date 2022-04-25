load('digits4000.mat')
warning('off');
% Preprocessing
trainX_1 = digits_vec(:,trainset(1,:))';
testX_1 = digits_vec(:,testset(1,:))';
trainY_1 = digits_labels(:,trainset(1,:));
testY_1 = digits_labels(:, testset(1,:));

% Training linear regression model
regression_model_1 = glmfit(trainX_1, trainY_1);
trainX_1 = [ones(2000, 1), trainX_1];
testX_1 = [ones(2000, 1), testX_1];

result = [];
for i = 1 : 2000
    temp = dot(regression_model_1', testX_1(i, :));
    result(i) = 1 / (1 + exp(-temp));
end

regression_model_1 = gradientDescent(trainX_1, result, regression_model_1, 0.05, 20);

result = [];
for i = 1 : 2000
    temp = dot(regression_model_1', testX_1(i, :));
    result(i) = 1 / (1 + exp(-temp));
end
% result = glmval(regression_model_1, testX_1, 'logit');
acc_1 = 0.;
for i = 1 : 2000
    if result(i) > 0.5
        acc_1 = acc_1 + 1;
    end
end
acc_1 = acc_1 / 2000;

% Trial 2
trainX_2 = digits_vec(:,trainset(2,:))';
testX_2 = digits_vec(:,testset(2,:))';
trainY_2 = digits_labels(:,trainset(2,:));
testY_2 = digits_labels(:, testset(2,:));

regression_model_2 = glmfit(trainX_2, trainY_2);
trainX_2 = [ones(2000, 1), trainX_2];
testX_2 = [ones(2000, 1), testX_2];

for i = 1 : 2000
    temp = dot(regression_model_2', testX_2(i, :));
    result(i) = 1 / (1 + exp(-temp));
end

regression_model_2 = gradientDescent(trainX_2, result, regression_model_2, 0.05, 20);

for i = 1 : 2000
    temp = dot(regression_model_2', testX_2(i, :));
    result(i) = 1 / (1 + exp(-temp));
end

acc_2 = 0.;
for i = 1 : 2000
    if result(i) > 0.5
        acc_2 = acc_2 + 1;
    end
end
acc_2 = acc_2 / 2000;
fprintf('Logistic Regression: \n');
fprintf('Test Accuracy of Trial 1：%f\n', acc_1);
fprintf('Test Accuracy of Trial 2：%f\n', acc_2);
fprintf('Mean Accuracy：%f\n', mean([acc_1, acc_2]));
fprintf('Standard deviation：%f\n', std([acc_1, acc_2]));
fprintf('\n');