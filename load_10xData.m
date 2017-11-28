function [sdata, geneID_map] = load_10xData(data_dir,varargin)
% This function will load scRNA-seq data output from the 10x Genomics
% Cell Ranger pipeline. data_dir should be the path to the directory 
% containing three files: barcodes.tsv, genes.tsv, and matrix.mtx.
% n_cells is the number of cells to randomly subsample 
% from the matrix (default 40,000 cells). 

tic

n_cells = Inf;
return_sparse = false;

if isempty(data_dir)
    data_dir = './';
elseif data_dir(end) ~= '/'
    data_dir = [data_dir '/']; 
end

for i=1:length(varargin)-1
    if (strcmp(varargin{i}, 'sparse'))
        return_sparse = varargin{i+1};
    end
    if (strcmp(varargin{i}, 'n_cells'))
        n_cells = varargin{i+1};
    end
end

filename_dataMatrix = [data_dir 'matrix.mtx'];
filename_genes = [data_dir 'genes.tsv'];
filename_cells = [data_dir 'barcodes.tsv'];


% Read in gene expression matrix (sparse matrix)
% Rows = genes, columns = cells
fprintf('LOADING\n')
dataMatrix = mmread(filename_dataMatrix);
fprintf('  Data matrix (%i cells x %i genes): %s\n', ...
        size(dataMatrix'), ['''' filename_dataMatrix '''' ])

% Read in row names (gene names / IDs)
dataMatrix_genes = table2cell( ...
                   readtable(filename_genes, ...
                             'FileType','text','ReadVariableNames',0));
dataMatrix_cells = table2cell( ...
                   readtable(filename_cells, ...
                             'FileType','text','ReadVariableNames',0));

                         
% Remove empty cells
col_keep = any(dataMatrix,1);
dataMatrix = dataMatrix(:,col_keep);
dataMatrix_cells = dataMatrix_cells(col_keep,:);
fprintf('  Removed %i empty cells\n', full(sum(~col_keep)))

% Subsample cells
if n_cells < size(dataMatrix,2)
    fprintf('  Subsample cells, N = %i\n',n_cells)
    col_keep = randsample(size(dataMatrix,2), n_cells);
    dataMatrix = dataMatrix(:,col_keep);
end

% Store gene name/ID map
geneID_map = containers.Map(dataMatrix_genes(:,1), dataMatrix_genes(:,2));

% Remove empty genes
genes_keep = any(dataMatrix,2);
dataMatrix = dataMatrix(genes_keep,:);
dataMatrix_genes = dataMatrix_genes(genes_keep,:);
fprintf('  Removed %i empty genes\n', full(sum(~genes_keep)))

% Convert to sdata object
% Rows = cells, columns = genes
if return_sparse
    sdata = struct();
    sdata.data = dataMatrix';
    sdata.genes = dataMatrix_genes(:,2);
    sdata.ENSIDs = dataMatrix_genes(:,1);
    sdata.cells = dataMatrix_cells(:,1);
    sdata.library_size = sum(dataMatrix,1)';
    sdata.name_channel_map = containers.Map(sdata.genes, ...
                                            1:length(sdata.genes));
else
    sdata = scRNA_data('data_matrix', full(dataMatrix'), ... 
                       'cell_names', dataMatrix_cells, ... 
                       'gene_names_ENS', dataMatrix_genes(:,1), ... 
                       'gene_names', dataMatrix_genes(:,2));
end
toc
fprintf('\n%i x %i (cells x genes) MATRIX\n', size(sdata.data))
end