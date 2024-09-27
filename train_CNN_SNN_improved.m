
%% Improved Training of CNN and SNN for Semantic Tumor Segmentation
clc
clear
close all

%% Training a CNN:

imageDir = fullfile('ref');
labelDir = fullfile('GT');

imds = imageDatastore(imageDir);

classNames = ["Cancer" "background"];
labelIDs = [255 0];
pxds = pixelLabelDatastore(labelDir, classNames, labelIDs);
ds = pixelLabelImageDatastore(imds, pxds);

options = trainingOptions('sgdm', ...
    'InitialLearnRate',1e-3, ...  % Adjusted learning rate for better convergence
    'MaxEpochs',100, ...  % Increased epochs for more robust training
    'LearnRateDropFactor',5e-1, ...  % Adjusted drop factor
    'LearnRateDropPeriod',25, ...  % Updated drop period for gradual decay
    'LearnRateSchedule','piecewise', ...
    'MiniBatchSize',32);  % Increased batch size

% Define the threshold for spiking (adaptive)
initialSpikeThreshold = 0.5;
spikeThresholdDecay = 0.001;  % Gradual decay of threshold during training

% Define CNN layers (improved architecture)
CNNlayers = [
    imageInputLayer([72 72 1])
    convolution2dLayer(3,32,'Padding',1)  % Increased number of filters
    batchNormalizationLayer  % Added batch normalization for stability
    reluLayer
    dropoutLayer(0.3)  % Reduced dropout rate for better generalization
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,64,'Padding',1)  % Increased filters in the second conv layer
    batchNormalizationLayer
    reluLayer
    transposedConv2dLayer(4,32,'Stride',2,'Cropping',1)
    convolution2dLayer(1,2)
    softmaxLayer
    dicePixelClassificationLayer('dice')];

% Training the improved CNN
CNNnet = trainNetwork(ds, CNNlayers, options);

% Save the CNN network
save('cnn_improved.mat', 'CNNnet');

% Conversion from CNN to SNN with adaptive spiking layers
SNNlayers = [
    imageInputLayer([72 72 1])
    convolution2dLayer(3,32,'Padding',1)
    batchNormalizationLayer
    SpikeConversionLayer('SpikeConversion1')
    SpikingLayer(initialSpikeThreshold, spikeThresholdDecay, 'SpikingLayer1')  % Adaptive threshold spiking
    dropoutLayer(0.3)
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,64,'Padding',1)
    batchNormalizationLayer
    SpikeConversionLayer('SpikeConversion2')
    SpikingLayer(initialSpikeThreshold, spikeThresholdDecay, 'SpikingLayer2')
    transposedConv2dLayer(4,32,'Stride',2,'Cropping',1)
    convolution2dLayer(1,2)
    SpikeConversionLayer('SpikeConversion3')
    SpikingLayer(initialSpikeThreshold, spikeThresholdDecay, 'SpikingLayer3')
    softmaxLayer
    dicePixelClassificationLayer('dice')];

% Training the improved SNN
SNNnet = trainNetwork(ds, SNNlayers, options);

% Save the SNN network
save('snn_improved.mat', 'SNNnet');

% Visualization and performance metrics for CNN and SNN
for i=1:1
    for j=1:26
        str1=['ref/',num2str(i) ,' (',num2str(j),').jpg'];
        im=imread(str1);
        str2=['GT/',num2str(i) ,' (',num2str(j),').jpg'];
        GT=imread(str2);
        subplot(1,3,1), imshow(im), title('Input (train sample)')
        subplot(1,3,2), imshow(GT), title('Ground Truth')

        GT(GT>0)=1;
        
        % CNN Results
        [C_CNN,scores_CNN] = semanticseg(im,CNNnet);
        B_CNN=(C_CNN=='Cancer');
        subplot(1,3,3), imshow(B_CNN), title('CNN result')
        
        % SNN Results
        [C_SNN,scores_SNN] = semanticseg(im,SNNnet);
        B_SNN=(C_SNN=='Cancer');
        subplot(1,3,3), imshow(B_SNN), title('SNN result')

        % Calculate accuracy and Qc for SNN
        nResult=sum(sum(B_SNN==1));
        nGT=sum(sum(GT==1));
        nUNI=0;
        for w=1:numel(GT)
            if B_SNN(w)==1 && GT(w)==1
                nUNI=nUNI+1;
            end
        end
        Qc = nUNI/nGT * nUNI/nResult;
        acc = sum(sum(B_SNN==logical(GT)))/numel(GT);

        disp(['Sample ', num2str(j), ' | Qc: ', num2str(Qc), ' | Accuracy: ', num2str(acc)]);
        pause;
    end
end
