classdef SpikeConversionLayer < nnet.layer.Layer
    % SpikeConversionLayer converts the CNN output to spiking inputs
    
    properties
        % Define properties if needed
    end
    
    methods
        function layer = SpikeConversionLayer(name)
            % Create a SpikeConversionLayer
            layer.Name = name;
            layer.Description = 'Converts CNN output to spiking inputs';
        end
        
        function Z = predict(layer, X)
            % Convert outputs to spiking format
            spikeThreshold = 0.5; % Define a suitable threshold for your use case
            Z = single(X > spikeThreshold); % Convert to single precision
        end
    end
end
