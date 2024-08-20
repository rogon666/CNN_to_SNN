%% Training a CNN and a SNN for semantic tumor segmentaion
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
 
k = 1;
disp('------------------------------------------')
disp('     Train results: CNN   ');
for i=1:1  % max=16
    for j=1:26
        str1=['ref/',num2str(i) ,' (',num2str(j),').jpg'];
        im=imread(str1);
        str2=['GT/',num2str(i) ,' (',num2str(j),').jpg'];
        GT=imread(str2);
        subplot(1,3,1), imshow(im), title('Input (train sample)')
        subplot(1,3,2), imshow(GT), title('Ground Truth')

        GT(GT>0)=1;
        [C,scores] = semanticseg(im,CNNnet);
        B=(C=='Cancer');
        
        nResult=sum(sum(B==1));
        nGT=sum(sum(GT==1));
        nUNI=0;
        for w=1:numel(GT)
            if B(w)==1 && GT(w)==1
                nUNI=nUNI+1;
            end
        end
        k
        Qc= nUNI/nGT * nUNI/nResult
        
        acc= sum(sum(B==logical(GT)))/numel(GT)
        accuracy(k)=acc;
        Q(k)=Qc;
        k=k+1;
        subplot(1,3,3), imshow(B), title('CNN result')
        pause;
    end
end

%% SNN (conversion from CNN to SNN):

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

disp('------------------------------------------')
disp('     Train results: SNN   ');
for i=1:1  % max=16
    for j=1:26
        str1=['ref/',num2str(i) ,' (',num2str(j),').jpg'];
        im=imread(str1);
        str2=['GT/',num2str(i) ,' (',num2str(j),').jpg'];
        GT=imread(str2);
        subplot(1,3,1), imshow(im), title('Input (train sample)')
        subplot(1,3,2), imshow(GT), title('Ground Truth')

        GT(GT>0)=1;
        [C,scores] = semanticseg(im,SNNnet);
        B=(C=='Cancer');

        nResult=sum(sum(B==1));
        nGT=sum(sum(GT==1));
        nUNI=0;
        for w=1:numel(GT)
            if B(w)==1 && GT(w)==1
                nUNI=nUNI+1;
            end
        end
        k
        Qc= nUNI/nGT * nUNI/nResult

        acc= sum(sum(B==logical(GT)))/numel(GT)
        accuracy(k)=acc;
        Q(k)=Qc;
        k=k+1;
        subplot(1,3,3), imshow(B), title('SNN result')
        pause;
    end
end

