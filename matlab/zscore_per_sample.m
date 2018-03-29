function data_new = zscore_per_sample(data, sample_vec)

data_new = data;
uniq_samples = unique(sample_vec);
for I=1:length(uniq_samples)
    I
    curr_sample = uniq_samples(I);
    curr_idx = sample_vec == curr_sample;
    data_new(curr_idx, :) = zscore(data_new(curr_idx, :));
end