%% Load bunny
G = gsp_bunny();
sig = sin(1:G.N);
gsp_plot_signal(G, 1:G.N);

%% Construct external measurements


%% Run phate and get kernel
[axy, gxy] = alphakernel(data);

%% Construct graph and do gft
G_ft = gsp_compute_fourier_basis(G); % Clear G_0 if data gets big
%G.coords = Y;
ft = gsp_gft(G_ft, sig);
figure;
subplot(1,2,1);gsp_plot_signal(G_ft,sig);
subplot(1,2,2);gsp_plot_signal_spectral(G_ft, ft);
ft(1:1200) = 0;
f = gsp_igft(G_ft, ft);
figure;gsp_plot_signal(G, f);

%% Now Try Magic 
t = 500;
data_features = G.coords;
data_imputed = run_magic([data_features, f'], t, 'npca', 4, 'k', 10, 'lib_size_norm', false);
[Y, diffOp, DiffOp_t] = phate(data_features, 'npca', 3, 'mds_method', 'cmds', 'ndim', 3);
figure;scatter3(Y(:, 1), Y(:, 2), Y(:, 3), 20, data_imputed(:,4), 'filled');


%% Plot Magic
marker_size = 20;
subplot(1,2,1);
scatter3(data_features(:,1), data_features(:,2), data_features(:,3), marker_size, data_imputed(:,4), 'filled');
subplot(1,2,2);
scatter3(data_imputed(:, 1), data_imputed(:,2), data_imputed(:, 3), marker_size, data_imputed(:,4), 'filled');
subplot(1,);


%% Smooth by gsp_design_heat i.e. similar to Magic
tau = 100;
G_smoothed = gsp_design_heat(G, tau);
wave_coefs = gsp_filter_analysis(G, G_smoothed, sig);

figure;
gsp_plot_signal(G, wave_coefs);

