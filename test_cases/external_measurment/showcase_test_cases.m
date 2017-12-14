num_points = 1000; noise = 0.05;
[X_swiss, l_swiss, t_swiss] = generate_data('swiss', num_points, noise);
% Broken swiss might be nice for batch correction?
[X_broken_swiss, l_broken_swiss, t_broken_swiss] = generate_data('brokenswiss', num_points, noise);
[X_changing_swiss, l_changing_swiss, t_changing_swiss] = generate_data('changing_swiss', num_points, noise);

marker_size = 15;
figure;title('Normal Swiss Roll');
scatter3(X_swiss(:,1), X_swiss(:,2), X_swiss(:,3), marker_size, l_swiss, 'filled');colormap jet;
figure;title('Broken Swiss Roll');
scatter3(X_broken_swiss(:,1), X_broken_swiss(:,2), X_broken_swiss(:,3), marker_size, l_broken_swiss, 'filled');colormap jet;
figure;title('Changing Swiss Roll i.e. increasing density');
scatter3(X_changing_swiss(:,1), X_changing_swiss(:,2), X_changing_swiss(:,3), marker_size, l_changing_swiss, 'filled');colormap jet;
