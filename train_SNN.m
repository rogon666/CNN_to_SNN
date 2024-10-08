%% Training a CNN and a SNN for semantic tumor segmentaion
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
ds = pixelLabelImageDatastore(imds, pxds);

options = trainingOptions('sgdm', ...
    'InitialLearnRate',1e-2, ...
    'MaxEpochs',50, ...
    'LearnRateDropFactor',1e-1, ...
    'LearnRateDropPeriod',50, ...
    'LearnRateSchedule','piecewise', ...
    'MiniBatchSize',16);

% Define the threshold for spiking
spikeThreshold = 0.5;

% Define CNN layers
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

% Training the original CNN
CNNnet = trainNetwork(ds, CNNlayers, options);

% Save the CNN network
save('cnn.mat', 'CNNnet');

% Convert CNN layers to spiking layers
SNNlayers = [
    imageInputLayer([72 72 1])
    convolution2dLayer(3,16,'Padding',1)
    SpikeConversionLayer('SpikeConversion1') % Added Spike Conversion Layer
    SpikingLayer(spikeThreshold, 'SpikingLayer1') % Added Spiking Layer
    dropoutLayer(0.5)
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,32,'Padding',1)
    SpikeConversionLayer('SpikeConversion2') % Added Spike Conversion Layer
    SpikingLayer(spikeThreshold, 'SpikingLayer2') % Added Spiking Layer
    transposedConv2dLayer(4,16,'Stride',2,'Cropping',1)
    convolution2dLayer(1,2)
    SpikeConversionLayer('SpikeConversion3') % Added Spike Conversion Layer
    SpikingLayer(spikeThreshold, 'SpikingLayer3') % Added Spiking Layer
    softmaxLayer
    dicePixelClassificationLayer('dice')];

% Training the modified SNN
SNNnet = trainNetwork(ds, SNNlayers, options);

% Save the SNN network
save('snn.mat', 'SNNnet');
