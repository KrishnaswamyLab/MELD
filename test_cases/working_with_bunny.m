%% Load bunny
G = gsp_bunny();
sig = sin(1:G.N);

%% Construct external measurements
% Heat Filter
ext_meas_heat = make_ext_meas(G);
ext_meas_heat_noised = awgn(ext_meas_heat,25);
figure;suptitle('Heat Filter Measurements with and without noise');
subplot(1,2,1);
gsp_plot_signal(G, ext_meas_heat);
subplot(1,2,2);
gsp_plot_signal(G, ext_meas_heat_noised);

% Abspline Filter
ext_meas_abspline = make_ext_meas(G, 'filter', 'abspline');
ext_meas_abspline_noised = awgn(ext_meas,20);
figure;suptitle('AB spline wavelet Filter Measurements with and without noise');
subplot(1,2,1);
gsp_plot_signal(G, ext_meas_abspline);
subplot(1,2,2);
gsp_plot_signal(G, ext_meas_abspline_noised);

%% Run phate to get diffusion operator
data = G.coords;
t = 20;
[~, ~, diffOp_t] = phate(data, 'npca', 3, 'mds_method', 'cmds', 't', t);

%% Plot Imputed measurement on the Bunny
% Heat
imputed_meas_heat = diffOp_t * ext_meas_heat_noised;
figure;suptitle('pre-Imputed and post-Imputed for Heat measurements');
subplot(1,2,1);
gsp_plot_signal(G, ext_meas_heat_noised);
subplot(1,2,2);
gsp_plot_signal(G, imputed_meas_heat);

% Wavelet AB spline 
imputed_meas_abspline = diffOp_t * ext_meas_abspline_noised;
figure;suptitle('pre-Imputed and post-Imputed for AB spline measurements');
subplot(1,2,1);
gsp_plot_signal(G, ext_meas_abspline_noised);
subplot(1,2,2);
gsp_plot_signal(G, imputed_meas_abspline);


