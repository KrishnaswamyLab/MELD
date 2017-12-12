function [M, genes_found, ind_found, ind] = get_channel_data(obj, channels, rel)
% [M, genes_found] = get_channel_data(obj, channels)

if isprop(obj, 'genes')
    all_channels = obj.genes;
else
    all_channels = obj.channel_name_map;
end

ind = ismember(lower(all_channels), lower(channels));
%ind = ~cellfun('isempty',strfind(lower(all_channels), lower(channels)));
genes_found = all_channels(ind);
[~,ind_found] = intersect(lower(channels), lower(genes_found));

if sum(ind)>1
    disp('more than one gene found!!!');
    %find(ind)
end

if exist('rel','var') && rel
    ls = sum(obj.data,2);
    M = obj.data ./ repmat(ls, 1, size(obj.data,2));
    M = M .* median(ls);
    M = M(:,ind);
else
    M = obj.data(:,ind);
end