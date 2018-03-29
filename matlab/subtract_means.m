function data_new = subtract_means(data, sample_vec)

[mu_mat, groups] = grpstats(data, sample_vec, {'mean' 'gname'});
groups
data_new = data;
uniq_samples = unique(sample_vec);
for I=1:length(uniq_samples)
    I
    curr_sample = uniq_samples(I);
    data_new(sample_vec == curr_sample, :) = data_new(sample_vec == curr_sample, :) ...
        - repmat(mu_mat(I,:),sum(sample_vec == curr_sample),1);
end