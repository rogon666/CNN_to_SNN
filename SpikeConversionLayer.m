classdef SpikeConversionLayer < nnet.layer.Layer ...
        & nnet.layer.Formattable
    %SPIKECONVERSIONLAYER Converts an input to a Poisson spike train.

    %   Copyright 2023 The MathWorks, Inc.

    properties
        % RescaleFactor
        % Used to control the number of spikes produced by the network by
        % rescaling the random matrix which is used to generate the spike
        % train.
        RescaleFactor
    end

    methods
        function layer = SpikeConversionLayer(rescaleFactor)
            %SPIKINGCONVERSIONLAYER Construct an instance of this class.
            arguments
                rescaleFactor (1,1) {mustBeNumeric}
            end

            % Set the layer description
            layer.Description = "Spiking conversion layer";

            % Set the layer rescale factor
            layer.RescaleFactor = rescaleFactor;
        end

        function Z = predict(layer,X)
            % PREDICT Convert an input to a poisson spike train.
            % spikeThreshold is used as a threshold to randomly
            % generate spikes.
            spikeThreshold = rand(size(X)) * layer.RescaleFactor;
            Z = cast(spikeThreshold <= X,like=X);
        end
    end
end