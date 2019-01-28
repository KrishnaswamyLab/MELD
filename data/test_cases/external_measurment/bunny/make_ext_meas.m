function [ ext_meas ] = make_ext_meas(G, varargin)

% OUTPUT
%       ext_meas = External Measurements for the (bunny) graph produced 
%                  with either heat filter or asbpline filter

% INPUT
%       G = Graph. Must be like graph data type found in GSP toolbox
%       Specifically defined for the gsp_bunny() graph. 
% varargin:
%   'Amp' (default = 10)
%       Amplitude of both dirac deltas.
%   'center1' (default = 800)
%       Location of first dirac delta. 
%   'center2' (default = 2500)
%       Location of first dirac delta.
%   'filter' (default = heat)
%       Choice of filter to be applied to both deltas to create 
%       the external measurement. 

% Default settings
Amp1 = 12;
Amp2 = 10;
center1 = 800;
center2 = 2500;
filter = 'heat';

% Get input parameters
for i=1:length(varargin)
    if(strcmp(varargin{i},'Amp1'))
        Amp1 =  lower(varargin{i+1});
    end
    if(strcmp(varargin{i},'Amp2'))
        Amp1 =  lower(varargin{i+1});
    end
    if(strcmp(varargin{i},'center1'))
        center1 =  lower(varargin{i+1});
    end
    if(strcmp(varargin{i},'center2'))
        center2 =  lower(varargin{i+1});
    end
    if(strcmp(varargin{i},'filter'))
        filter =  lower(varargin{i+1});
    end
end


switch filter
    case 'heat'
        delta = zeros(1,G.N);
        delta(:,center1) = Amp1;
        tau = 100;
        heat = gsp_design_heat(G,tau);
        delta_heat = gsp_filter_analysis(G, heat, delta');
        
        delta2 = zeros(1,G.N);
        delta2(:,center2) = Amp2;
        tau = 100;
        heat2 = gsp_design_heat(G,tau);
        delta_heat2 = gsp_filter_analysis(G, heat2, delta2');
        ext_meas = delta_heat + delta_heat2;
        
    case 'abspline'
        Nf = 4; % Number of filters, we choose the second one most different from heat
        delta = zeros(1,G.N);
        delta(:,center1) = Amp1;
        abspline = gsp_design_abspline(G, Nf);
        delta_abspline = gsp_filter_analysis(G, abspline{2,1}, delta');
        
        delta2 = zeros(1,G.N);
        delta2(:,center2) = Amp2;
        abspline2 = gsp_design_abspline(G, Nf);
        delta_abspline2 = gsp_filter_analysis(G, abspline2{2,1}, delta2');
        ext_meas = delta_abspline + delta_abspline2;

end

