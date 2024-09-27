%% Training a CNN for semantic tumor segmentaion
clc
clear
close all
% Train
imageDir = fullfile('ref');
labelDir = fullfile('GT');

imds = imageDatastore(imageDir);

classNames = ["Cancer" "background"];
labelIDs = [255 0];
pxds = pixelLabelDatastore(labelDir, classNames, labelIDs);
ds = pixelLabelImageDatastore(imds,pxds);

options = trainingOptions('sgdm', ...
    'InitialLearnRate',1e-2, ...
    'MaxEpochs',150, ...
    'LearnRateDropFactor',1e-1, ...
    'LearnRateDropPeriod',50, ...
    'LearnRateSchedule','piecewise', ...
    'MiniBatchSize',16);
layers = [
    imageInputLayer([72 72 1])
    convolution2dLayer(3,16,'Padding',1)
    reluLayer
    dropoutLayer(0.5)
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,32,'Padding',1)
    reluLayer
    transposedConv2dLayer(4,16,'Stride',2,'Cropping',1)
    convolution2dLayer(1,2)
    softmaxLayer
    dicePixelClassificationLayer('dice')];
net = trainNetwork(ds,layers,options);
save('cnn.mat','net');
