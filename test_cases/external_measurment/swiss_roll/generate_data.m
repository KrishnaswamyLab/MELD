function [X, labels, t] = generate_data(dataname, n, noise, a, b)
%GENERATE_DATA Generates an artificial dataset (manifold)
%
%	[X, labels, t] = generate_data(dataname, n, noise)
%
% Generates an artificial dataset. Possible datasets are: 'swiss' for the Swiss roll
% dataset. The variable n indicates the number of datapoints to generate 
% (default = 1000). The variable noise indicates the amount of noise that
% is added to the data (default = 0.05). The function returns the
% high-dimensional dataset in X, and corresponding labels in labels. In
% addition, the function returns the coordinates of the datapoints on the
% underlying manifold in t.
%
%

% This file is part of the Matlab Toolbox for Dimensionality Reduction.
% The toolbox can be obtained from http://homepage.tudelft.nl/19j49
% You are free to use, change, or redistribute this code in any way you
% want for non-commercial purposes. However, it is appreciated if you 
% maintain the name of the original author.
%
% (C) Laurens van der Maaten, Delft University of Technology

% Shamelessly gotten from van der Maaten
    
	if ~exist('n', 'var')
		n = 1000;
    end
    if ~exist('noise', 'var')
        noise = 0.05;
    end

	switch dataname
        case 'swiss'
            t = (3 * pi / 2) * (1 + 2 * rand(n, 1));  
            height = 30 * rand(n, 1);
            X = [t .* cos(t) height t .* sin(t)] + noise * randn(n, 3);
            %labels = uint8(t);
            %labels = rem(sum([round(t / a) round(height / b)], 2), 2);
            labels = 25*sin(t./a + height ./ b);
            t = [t height];
            
        case 'brokenswiss'
            t = [(3 * pi / 2) * (1 + 2 * rand(ceil(n / 2), 1) * .4); (3 * pi / 2) * (1 + 2 * (rand(floor(n / 2), 1) * .4 + .6))];  
            height = 30 * rand(n, 1);
            X = [t .* cos(t) height t .* sin(t)] + noise * randn(n, 3);
            labels = uint8(t);
            %labels = rem(sum([round(t / 2) round(height / 12)], 2), 2);
            t = [t height];
            
        case 'changing_swiss'
            r = zeros(1, n);
            for i=1:n
                pass = 0;
                while ~pass
                    rr = rand(1);
                    if rand(1) > rr
                        r(i) = rr;
                        pass = 1;
                    end
                end
            end
            t = (3 * pi / 2) * (1 + 2 * r);  
            height = 21 * rand(1, n);
            X = [t .* cos(t); height; t .* sin(t)]' + noise * randn(n, 3);
            %labels = uint8(t)';
            labels = rem(sum([round(t / 2); round(height / 10)], 1), 2)';
            
        case 'helix'
        	t = [1:n]' / n;
        	t = t .^ (1.0) * 2 * pi;
			X = [(2 + cos(8 * t)) .* cos(t) (2 + cos(8 * t)) .* sin(t) sin(8 * t)] + noise * randn(n, 3);
        	%labels = uint8(t);
            labels = rem(round(t * 1.5), 2);

		otherwise
			error('Unknown dataset name.');
	end
