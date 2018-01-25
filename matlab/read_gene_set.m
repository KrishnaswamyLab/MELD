function gene_set = read_gene_set(file)

fid = fopen(file);
gene_set = textscan(fid, '%s', 'Delimiter', '\n');
fclose(fid);

if strcmp(gene_set{1}{2}(1),'>')
    disp 'file has header:'
    gene_set{1}(1:2)
    gene_set = gene_set{1}(3:end);
else
    gene_set = gene_set{1};
end