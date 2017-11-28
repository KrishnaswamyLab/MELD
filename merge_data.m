function [sdata, sample_vec] = merge_data(sdata_vec, sample_names)

sdata_A = sdata_vec{1};
sample_vec = ones(size(sdata_A.data,1),1);

for I=2:length(sdata_vec)
    I
    sdata_B = sdata_vec{I};
    ncells = size(sdata_A.data,1) + size(sdata_B.data,1);
    sdata = sdata_A;
    ENSIDs_vec = [sdata_A.ENSIDs; sdata_B.ENSIDs];
    genes_vec = [sdata_A.genes; sdata_B.genes];
    [ENSIDs_vec, uind] = unique(ENSIDs_vec);
    genes_vec = genes_vec(uind);
    sdata.ENSIDs = ENSIDs_vec;
    sdata.genes = genes_vec;
    sdata.data = zeros(ncells, length(sdata.ENSIDs));
    [~,IA,IB] = intersect(sdata_A.ENSIDs, sdata.ENSIDs);
    sdata.data(1:size(sdata_A.data,1), IB) = sdata_A.data(:,IA);
    [~,IA,IB] = intersect(sdata_B.ENSIDs, sdata.ENSIDs);
    sdata.data(size(sdata_A.data,1)+1:end, IB) = sdata_B.data(:,IA);
    sdata.library_size = [sdata_A.library_size; sdata_B.library_size];
    sdata.cells = [sdata_A.cells; sdata_B.cells];
    sdata_A = sdata;
    sample_vec = [sample_vec; I+zeros(size(sdata_B.data,1),1)];
end

sdata.mpg = sum(sdata.data);
sdata.cpg = sum(sdata.data > 0);
sdata.samples = sample_vec;
sdata.sample_names = sample_names;
sdata = sdata.recompute_name_channel_map()