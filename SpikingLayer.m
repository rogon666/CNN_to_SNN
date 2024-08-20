classdef SpikingLayer < nnet.layer.Layer
    properties
        % Define any properties needed for the layer
        Threshold
    end
    
    methods
        function layer = SpikingLayer(threshold, name)
            % Create a spiking layer with the specified threshold
            layer.Name = name;
            layer.Description = "Spiking layer with threshold " + threshold;
            layer.Threshold = threshold;
        end
        
        function Z = predict(layer, X)
            % Forward input data through the layer at prediction time
            Z = single(X > layer.Threshold);
        end
    end
end
