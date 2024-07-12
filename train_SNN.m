%% Training a SNN for semantic tumor segmentaion
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
CNNlayers = [
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
CNNnet = trainNetwork(ds,CNNlayers,options);

% Convert CNN to SNN
spikeThreshold = 0.5;

% Remove the output layers for dlnetwork
snnLayersForDLNetwork = [
    CNNnet.Layers(1) % imageInputLayer
    SpikeConversionLayer(spikeThreshold) % Convert input to spikes
    SpikingLayer(CNNnet.Layers(2)) % convolution2dLayer
    CNNnet.Layers(3) % reluLayer (no need for spiking behavior)
    CNNnet.Layers(4) % dropoutLayer (no need for spiking behavior)
    CNNnet.Layers(5) % maxPooling2dLayer (no need for spiking behavior)
    SpikingLayer(CNNnet.Layers(6)) % convolution2dLayer
    CNNnet.Layers(7) % reluLayer (no need for spiking behavior)
    SpikingLayer(CNNnet.Layers(8), 16) % transposedConv2dLayer
    SpikingLayer(CNNnet.Layers(9), 2)  % convolution2dLayer
    ];

net = dlnetwork(snnLayersForDLNetwork);
save('snn.mat','net');
