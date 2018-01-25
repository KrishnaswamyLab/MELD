a = 10;b = 20;
[Y,labs,~] = generate_data('swiss', 2000, 0.0, a, b);
marker_size = 50;
t = 50;
data_imputed = run_magic([Y,labs], t, 'npca', 3, 'k', 15, 'lib_size_norm', false);

figure;
scatter3(Y(:,1),Y(:,2),Y(:,3), marker_size, labs, 'filled');
title('Before Magic, with original external variable');

figure;
scatter3(Y(:,1),Y(:,2),Y(:,3), marker_size, data_imputed(:,4), 'filled');
title('Colored by Magic fourth eigenvector');

figure;
scatter3(data_imputed(:,1),data_imputed(:,2),data_imputed(:,3),marker_size,labs, 'filled');


