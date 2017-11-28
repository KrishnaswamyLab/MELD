%How do I extend this to time-series data?
%Need help extending this to time series. I need to make a struct of data
%qualities of time series data is that there arent equally as many cells


classdef scRNA_data
    
    properties
        
        %basic data
        data
        data_nn
        genes
        ENSIDs
        cells
        samples
        name
        name_channel_map
        fano_tech
        fano_bio
        cv_extr
        tsne_mapped_data
        experiment
        
        knnNd_idx;
        knnNd_dist;
        
        library_size;
        mpg;
        cpg;
        sample_names;
    end
    
    methods
        %input output
        function obj = scRNA_data(varargin)
            file_name = [];
            data_matrix = [];
            gene_names = [];
            gene_names_ENS = [];
            cell_names = [];
            samples = [];
            sample_names = [];
            delimiter = '\t';
            read_cell_labels = true;
            N_col_label = 1;
            for i=1:length(varargin)-1
                if(strcmp(varargin{i},'file_name'))
                    file_name = varargin{i+1};
                end
                if(strcmp(varargin{i},'data_matrix'))
                    data_matrix = varargin{i+1};
                end
                if(strcmp(varargin{i},'gene_names'))
                    gene_names = varargin{i+1};
                end
                if(strcmp(varargin{i},'gene_names_ENS'))
                    gene_names_ENS = varargin{i+1};
                end
                if(strcmp(varargin{i},'cell_names'))
                    cell_names = varargin{i+1};
                end
                if(strcmp(varargin{i},'samples'))
                    samples = varargin{i+1};
                end
                if(strcmp(varargin{i},'sample_names'))
                    sample_names = varargin{i+1};
                end
                if(strcmp(varargin{i},'delimiter'))
                    delimiter = varargin{i+1};
                end
                if(strcmp(varargin{i},'read_cell_labels'))
                    read_cell_labels = varargin{i+1};
                end
                if(strcmp(varargin{i},'N_col_label'))
                    N_col_label = varargin{i+1};
                end
            end
            if(~isempty(file_name))
                fid = fopen(file_name);
                line1 = strsplit(fgetl(fid),delimiter);
                ncol = length(line1);
                format = [repmat('%s',1,N_col_label) repmat('%f',1,ncol-N_col_label)];
                if read_cell_labels
                    obj.cells = line1(N_col_label+1:end)';
                    file_data = textscan(fid, format, 'delimiter', delimiter);
                else
                    fclose(fid);
                    fid = fopen(file_name);
                    file_data = textscan(fid, format, 'delimiter', delimiter);
                end
                fclose(fid);
                obj.genes = file_data{1};
                obj.data = cell2mat(file_data((N_col_label+1):end));
                obj.data = transpose(obj.data);
            end
            if(~isempty(data_matrix))
                obj.data = data_matrix;
            end
            if(~isempty(gene_names))
                obj.genes = gene_names;
            end
            if(~isempty(gene_names_ENS))
                obj.ENSIDs = gene_names_ENS;
            end
            if(~isempty(cell_names))
                obj.cells = cell_names;
            end
            if(~isempty(samples))
                obj.samples = samples;
            end
            if(~isempty(sample_names))
                obj.samples = sample_names;
            end
            obj.name_channel_map = containers.Map();
            for i=1:length(obj.genes)
                obj.name_channel_map(obj.genes{i}) = i;
            end
            
            
            obj.library_size = sum(obj.data,2);
            obj.mpg = sum(obj.data);
            obj.cpg = sum(obj.data > 0);
            
        end
        
        function obj = recompute_name_channel_map(obj)
            
           obj.name_channel_map = containers.Map();
            for i=1:length(obj.genes)
                obj.name_channel_map(obj.genes{i}) = i;
            end 
            
        end
        
        function obj = add_pseudogene(obj, gene_name, gene_data)
           
            obj.genes{end+1} = gene_name;
            gene_idx = length(obj.genes);
            obj.name_channel_map(gene_name) = gene_idx;
            obj.data(:,end+1) = gene_data;
            
        end
        
        function obj = normalize_data(obj)
%            data_norm = obj.data';
%             libsize = sum(data_norm);
%             data_norm = data_norm + 1;
%             data_norm = bsxfun(@rdivide, data_norm, libsize) .* mean(libsize);
%             data_norm = data_norm - 1;
%              obj.data = data_norm';           

            data_norm = transpose(obj.data);
            data_norm = bsxfun(@rdivide, data_norm, sum(data_norm)) .* mean(sum(data_norm));
            [m,n] = size(data_norm);
            jitter_values = abs(normrnd(.1,.01,m,n)) .* (data_norm == 0);
            data_norm = data_norm + jitter_values;
            obj.data = transpose(data_norm);

        end
        
        function write_cluster_genes(obj, cluster_vector, gene_names)
            
            num_clusters = max(cluster_vector);
            
            for i=1:num_clusters
                
                filename = sprintf('cluster%d.txt',i);
                file = fopen(filename,'w');
            
                cluster_indices = find(cluster_vector==i);
                for j=1:length(cluster_indices)
                    
                    genestring = gene_names{cluster_indices(j)};
                    fprintf(file,'%s\n', genestring);
                end
                fclose(file);
            
            end
        end
        

        function [gene_gene_mi_matrix, data_submatrix, genes] = compute_gene_gene_adjacency(obj, start_percentile, end_percentile)
            
            mean_gene_abundance = mean(obj.data,1);
            x = prctile(mean_gene_abundance, [start_percentile end_percentile])
            ind = find((mean_gene_abundance > x(1)) & (mean_gene_abundance < x(2)));
            genes = obj.genes(ind);
            num_genes = length(genes)
            data_submatrix = obj.data(:,ind);
            gene_gene_mi_matrix = zeros(length(genes),length(genes));
            
            for i=1:length(genes)
                i
                for j = 1:length(genes)
                    
                    if(i==j)
                        continue;
                    end
                    
                    %[gene_gene_mi_matrix(i,j), ~] = knn_mutual_information(obj, genes{i}, genes{j},'k_density',10);
                                       R = corrcoef(obj.data(:,i), obj.data(:,j));
                                       gene_gene_mi_matrix(i,j) = abs(R(1,2));
                    
                end
            end
           
        end
        
        function obj = normalize_data_fix_zero(obj, m)
            
            if ~exist('m','var')
                m = 0;
            end
            
            data_norm = obj.data';
            libsize = sum(data_norm);
            ind_zero = data_norm == 0;
            data_norm(ind_zero) = m;
            data_norm = bsxfun(@rdivide, data_norm, libsize) .* median(libsize);
            data_norm(ind_zero) = data_norm(ind_zero) - m;
            obj.data = data_norm';
        end
        
        function obj = throw_channel_zeros(obj, gene_name)
            
            channel = obj.name_channel_map(gene_name);
            throwpts = (obj.data(:,channel)==0);
            obj.data(throwpts,:)=[];
        end
            
       function [obj] = compute_outliers(obj,k, indegree_t)
        
        %eliminate outliers 
          
          %[knn_idx, knn_dist] = knnsearch(pc, pc,'K',k+1);
          for i=1:num_cells
              indegree(knn_idx(i,:)) = indegree(knn_idx(i,:)) + 1;
          end
            
           points = find(indegree>indegree_t);
           obj.data = obj.data(points,:);
           
          
            
       end
        
               %estimate density 

        
               %estimate density 
        function f = laplacian_density(~, varargin)
            L = [];
            t = 1;
            k = 1;
            for i=1:length(varargin)-1
                if(strcmp(varargin{i},'t'))
                    t = varargin{i+1};
                end
                if(strcmp(varargin{i},'L'))
                    L = varargin{i+1};
                end
                if(strcmp(varargin{i},'k_density'))
                    k = varargin{i+1};
                end
            end
            L = L^t;
            [~, knn_dist] = knnsearch(L, L, 'K', k+1);
            f = 1 ./ knn_dist(:,k+1);
        end
 
        function plot_2d_density(obj, channel1_name, channel2_name, k, point_densities)
            
            channel1 = obj.name_channel_map(channel1_name);
            channel2 = obj.name_channel_map(channel2_name);
            
          
            colormap(jet);
            ca = [1 99];
           
         
            scatter(obj.data(:,channel1), obj.data(:,channel2), 50, point_densities, 'fill');
            caxis(prctile(point_densities,ca));
            xlabel(channel1_name);
            ylabel(channel1_name);
            
          
        end
        

        function channels =  get_channels(obj, channel_names)
            
           
            channels = zeros(length(channel_names),1);
                    
               for i=1:length(channel_names)
                       
                   channels(i) = obj.name_channel_map(channel_names{i});
               end
            
        end
        
       function plot_2d_conditional_density(obj, channel1_name, channel2_name, k)
            
            [point_conditional_densities, rescaled_conditional_densities] = knn_conditional_density(obj, {channel1_name, channel2_name}, {channel1_name}, k);
            channel1 = obj.name_channel_map(channel1_name);
            channel2 = obj.name_channel_map(channel2_name);
            colormap(jet);
            scatter(obj.data(:,channel1), obj.data(:,channel2), 50, rescaled_conditional_densities, 'fill');
            title('recaled conditional density');
           
            
       end
       
       function [mi_matrix] = compute_mi_matrix(obj, genes, varargin)
           
           
           mi_matrix = zeros(length(genes), length(genes));
           
            for i=1:length(genes)
                
                for j=1:length(genes)
                    
                    
                    if(i>=j) 
                        continue;
                    end
                    
                    [mi_matrix(i,j)] = knn_mutual_information(obj, genes{i}, genes{j}, varargin{:});
                end
                
            end
           
       end
       
        function obj = fit_noise(obj, nbins, draw_plots, data)
            
         
            if ~exist('nbins', 'var')
                nbins = 100;
            end
            
            if ~exist('draw_plots', 'var')
                draw_plots = true;
            end
            
            if exist('data', 'var')
                mu = mean(data);
                sigmasq = var(data);
            else
                mu = mean(obj.data);
                sigmasq = var(obj.data);
            end
            fano = sigmasq ./ mu;
            
            ind_keep = mu ~= 0;
            fano = fano(ind_keep);
            mu = mu(ind_keep);
            
            [~,bin_centers] = hist(log(mu), nbins);
            bin_width = bin_centers(2) - bin_centers(1);
            bin_edges = [bin_centers + bin_width/2, inf];
            [~,bin_idx] = histc(log(mu), bin_edges); % convert centers to edges first
            
            fano_min = nan(size(bin_edges));
            
            for I=1:nbins
                fano_bin = log(fano(bin_idx == I-1));
                if ~isempty(fano_bin)
                    fano_min(I) = min(fano_bin);
                end
            end
            
            
            log_x = [bin_centers, bin_centers(end) + bin_width/2];
            log_y = fano_min;
            
            ind_keep = ~isnan(log_y);
            log_x = log_x(ind_keep);
            log_y = log_y(ind_keep);
            
            f = @(c,mu,fano) sum(abs(fano - ((1./mu + c) .* mu)) .* (1:length(mu)).^2);
            fun = @(x) f(x, exp(log_x), exp(log_y));
            C = fmincon(fun,0,[],[],[],[],0,1);
            
            %log_fano_fit = log((1./exp(log_x) + C) .* exp(log_x));
            
            obj.fano_tech = (1./mu + C) .* mu;
            obj.fano_bio = fano - obj.fano_tech;
            
            obj.cv_extr = sqrt(C);
            
            if draw_plots
                figure;
                subplot(2,2,1);
                hold on;
                plot(log(mu), log(fano), '.k');
                xlabel 'log mean'
                ylabel 'log fano total'
                
                subplot(2,2,2);
                hold on;
                plot(log(mu), log(fano), '.k');
                plot(log_x, log_y, '.r');
                %plot(log_x, log_fano_fit, '.g');
                plot(xlim, zeros(size(xlim)), '--b');
                plot(log(mu), log(obj.fano_tech), '.m');
                xlabel 'log mean'
                ylabel 'log fano total'
                title(['CV extr = ' num2str(sqrt(C)*100,2) '%']);
                
                %                 subplot(2,2,3);
                %                 dscatter(log(mu(obj.fano_bio>0)), log(obj.fano_bio(obj.fano_bio>0)));
                %                 xlabel 'log mean'
                %                 ylabel 'log fano bio'
                %                 title(['CV extr = ' num2str(sqrt(C)*100,2) '%']);
                
                subplot(2,2,3);
                plot(log(mu(obj.fano_bio>0)), log(obj.fano_bio(obj.fano_bio>0)), '.k');
                xlabel 'log mean'
                ylabel 'log fano bio'
                title(['CV extr = ' num2str(sqrt(C)*100,2) '%']);
                hold on;
                
                subplot(2,2,4);
                hold on;
                plot(log(mu), log(fano) - log(mu), '.k');
                plot(log_x, log_y - log_x, '.r');
                %plot(log_x, log_fano_fit - log_x, '.g');
                plot(xlim, -xlim, '--b');
                plot(log(mu), log(obj.fano_tech)-log(mu), '.m');
                xlabel 'log mean'
                ylabel 'log CV^2 total'
                title(['CV extr = ' num2str(sqrt(C)*100,2) '%']);
            end
        end
        
        function plot_2d_channel_hist(obj, channel1_name, channel2_name,nbins)
            
            
            channel1 = obj.name_channel_map(channel1_name);
            channel2 = obj.name_channel_map(channel2_name);
            data = [obj.data(:,channel1) obj.data(:,channel2)];
            
            M = hist3(data,nbins);
            imagesc(M);
            set(gca,'YDir','normal');
            xlabel(channel1_name);
            ylabel(channel2_name);
            colorbar;
        end
        
        function plot_channel_hist(obj, channel_name, num_bins, varargin)
            
            channel = obj.name_channel_map(channel_name);
            data = obj.data(:,channel);
            optargin = size(varargin,2);
            h = findobj(gca,'Type','patch');
            
            if(optargin==1)
                set(h,'FaceColor',varargin{1},'EdgeColor','w');
                hist(data, num_bins);
            else
                set(h,'FaceColor', 'b','EdgeColor','w');
                hist(data, num_bins);
            end
            
        end
        
        function [F, XI] = plot_channel_density(obj, channel_name, varargin)
            
            XI_given = 0;
            XI = [];
            for i=1:length(varargin)
                if(strcmp(varargin{i},'XI'))
                    XI_given = 1;
                    XI = varargin{i+1};
                end
            end
            
            channel = obj.name_channel_map(channel_name);
            
            if(XI_given==0)
                [F, XI] = ksdensity(obj.data(:,channel));
            else
                [F] = ksdensity(obj.data(:,channel), XI);
                
            end
            
            
            plot(XI,F,'LineWidth',5);
            
        end
        
        function plot_2d_channel_density(obj, channel1_name, channel2_name, varargin)
            
            figure;
            
            channel1 = obj.name_channel_map(channel1_name);
            channel2 = obj.name_channel_map(channel2_name);
            
            data = [obj.data(:,channel1) obj.data(:,channel2)];
            limits = [];
            for i=1:length(varargin)
                
                
                if(strcmp(varargin{i}, 'limits'))
                    
                    limits = varargin{i+1};
                    
                end
            end
            
            maxx = max(obj.data(:,channel1));
            
            if(length(limits)>0)
                
                [bandwidth,density,X,Y]=kde2d(data,256, [limits(1) limits(2)], [limits(3) limits(4)]);
            else
                
                [bandwidth,density,X,Y]=kde2d(data,256);
            end
            
            
            slices_size = maxx/8;
            
            optargin = length(varargin);
            if(optargin == 0)
                plot(data(:,1),data(:,2),'b.','MarkerSize',5)
                hold on,
                contour(X,Y,density,30),
            end
            
            
            for i=1:length(varargin)
                
                
                if(strcmp(varargin{i}, 'imagesc'))
                    
                    colormap(jet);
                    imagesc(X(1,:),Y(:,1),density);
                    set(gca,'YDir','normal');
             
                    
                end
                if(strcmp(varargin{i}, 'contour'))
                    
                    plot(data(:,1),data(:,2),'b.','MarkerSize',5)
                    hold on,
                    contour(X,Y,density,30),
                    set(gca,'XLim',[0 max(data(:,1))]);
                    set(gca,'YLim',[0 max(data(:,2))]);
                end
                
                
                
            end
            
            xlabel(channel1_name);
            ylabel(channel2_name);
            
        end
        
        function plot_2d_scatter(obj,channel1_name, channel2_name)
            
            channel1 = obj.name_channel_map(channel1_name);
            channel2 = obj.name_channel_map(channel2_name);
            
            plot(obj.data(:,channel1),obj.data(:,channel2),'*');
            
        end
        
        function draw_tsne(obj,varargin)
            
            add_color = false;
            color_channel_data = [];
            
            for i=1:length(varargin)
                
                if(strcmp(varargin{i},'color_data'))
            
                   add_color = true; 
                   color_channel_data = varargin{i+1};
                   
                end    
                if(strcmp(varargin{i},'color_channel'))
                    
                    add_color = true;
                    coloring_channel = varargin{i+1};
                    channel = obj.name_channel_map(coloring_channel);
                    color_channel_data = obj.data(:,channel);
                end
                
            end
            
            if(add_color)
                
                scatter(obj.tsne_mapped_data(:,1), obj.tsne_mapped_data(:,2), 50, color_channel_data, 'fill');
                ca = prctile(color_channel_data,[5 95]);
                caxis(ca);
                colormap(jet);
                
            else
                
                scatter(obj.tsne_mapped_data(:,1),obj.tsne_mapped_data(:,2),'.k');
                
            end
            
            
        end
        
        function obj =  tsne_map_data(obj,  channel_names)
            
            
            
            num_events = size(obj.data,1);
            
            channels=[];
            
            for i=1:length(channel_names)
                
                channels = [channels obj.name_channel_map(channel_names{i})];
                
            end
            %
            
            if false
                obj.tsne_mapped_data = fast_tsne(obj.data(:,channels));
            else
                disp 'debug: run t-sne on correlation matrix'
                disp 'computing pdist'
                D = squareform(pdist(obj.data(:,channels), 'correlation'));
                disp 'running t-sne'
                obj.tsne_mapped_data = fast_tsne(D);
            end
            
        end
        
        
        
        function [C_xy] = compute_causal_scores(obj, channel1_name, channel2_name, method_to_use)
            
            
            
            
            C_xy = 0;
            
            if(strcmp(method_to_use,'DREMI_Residual'))
                
                dremi1 = obj.compute_dremi(channel1_name, channel2_name,.8,8);
                dremi2 = obj.compute_dremi(channel2_name, channel1_name,.8,8);
                %actually use DREMI score under certain conditions.
                if(dremi1>dremi2)
                    
                    if(dremi2<=.15)
                        
                        if((dremi1-dremi2)>0.5)
                            C_xy = 1;
                            
                            return;
                        end
                    end
                end
                
                if(dremi2>dremi1)
                    
                    if(dremi1<=.15)
                        
                        if((dremi2-dremi1)>0.5)
                            C_xy =-1;
                            
                            return;
                        end
                    end
                end
                
                noise_dremi1 =  obj.compute_noise_dremi(channel1_name, channel2_name);
                noise_dremi2 = obj.compute_noise_dremi(channel2_name, channel1_name);
                %remember decision is backwards
                if(noise_dremi1>noise_dremi2)
                    C_xy = -1;
                    
                else
                    C_xy = 1;
                    
                end
            end
            
            if(strcmp(method_to_use,'DREMI'))
                
                dremi1 =  obj.compute_dremi(channel1_name, channel2_name,0.8);
                dremi2 =  obj.compute_dremi(channel2_name, channel1_name,0.8);
                if(dremi1>dremi2)
                    C_xy = 1;
                    
                else
                    C_xy = -1;
                    
                end
                
            end
            
            
            
        end
        
        
        
        function compute_and_graph_render_edges_dremi(obj, edges, nodes, varargin)
            
            ctrl_specificed = 0;
            ctrl_data = [];
            
            for i=1:length(varargin)
                
                if(strcmp(varargin{i}, 'ctrl_data'))
                    
                    ctrl_specified = 1;
                    ctrl_data = varargin{i+1};
                    
                    
                end
            end
            
            adjMatrix = zeros(length(nodes), length(nodes));
            edges_indexed = zeros(length(edges),2);
            for i = 1:length(edges)
                
                node_index1 = find(strcmp(edges{i,1},nodes));
                node_index2 = find(strcmp(edges{i,2},nodes));
                adjMatrix(node_index1, node_index2) = 1;
                edges_indexed(i,1) = node_index1;
                edges_indexed(i,2) = node_index2;
                
                
            end
            gObj = biograph(adjMatrix,nodes);
            
            
            for i=1:length(edges)
                
                
                if(ctrl_specified==0)
                    dremi_values(i) = obj.compute_dremi(edges{i,1},edges{i,2},.80);
                else
                    
                    channel1_name = edges{i,1};
                    channel2_name = edges{i,2};
                    
                    
                    [dremi_values(i),~] = obj.compute_dremi(channel1_name, channel2_name, noise_threshold);
                end
            end
            
            range = .65;
            load 'continuous_BuPu9.mat';
            colormap(continuous_BuPu9);
            
            
            
            for i = 1:length(dremi_values)
                
                [color_value] = get_color_value(dremi_values(i), .65, continuous_BuPu9);
                set(gObj.edges(i),'LineColor',color_value);
                set(gObj.edges(i),'LineWidth',2.0);
            end
            
            all_nodes = 1:length(nodes);
            set(gObj.nodes(all_nodes),'LineColor',[.4 .4 .4]);
            set(gObj.nodes(all_nodes),'Color',[1 1 1]);
            set(gObj.nodes(all_nodes),'Shape','ellipse');
            set(gObj.nodes(all_nodes),'LineWidth',1.1);
            set(gObj.nodes(all_nodes),'fontSize',14);
            view(gObj);
            
        end
        
        
        function [obj, mi_matrix ] = pairwise_dremi_compute(obj, channel_names, noise_threshold, varargin)
            
            
            mi_matrix = zeros(length(channel_names), length(channel_names));
            
            for i=1:length(channel_names)
                for j=1:length(channel_names)
                    if(i==j)
                        continue;
                    end
                    channel1_name = channel_names{i}
                    channel2_name = channel_names{j}
                    
                    [mi_matrix(i,j),~] = obj.compute_dremi(channel1_name, channel2_name, noise_threshold, 8);
                    
                end
            end
            
            obj.DREMI_adjacency = mi_matrix;
            
            for i=1:length(varargin)
                
                if(strcmp(varargin{i}, 'plot'))
                    
                    
                    colormap(jet)
                    CLIM = [0 1];
                    imagesc(mi_matrix);
                    set(gca,'ytick',1:length(channel_names));
                    set(gca,'yticklabel',channel_names);
                    xticklabel_rotate([1:length(channel_names)],45,channel_names);
                    colorbar
                    
                end
            end
            
            
        end
        
        
        
        function [DREMI ] = compute_dremi(obj, channel1_name, channel2_name, prob_threshold, num_partitions, varargin)
            
            compute_drevi = 1;
            
            for i=1:length(varargin)
                
                if(strcmp('drevi_matrix',varargin{1}))
                    
                    compute_drevi = 0;
                    DREVI = varargin{2};
                    
                end
            end
            
            if(compute_drevi==1)
                [DREVI] =  obj.compute_drevi( channel1_name, channel2_name,varargin,'no_plot');
            end
            
            
            [x_length, y_length] = size(DREVI);
            
            total_entropy = compute_sample_entropy_2d(DREVI, num_partitions, prob_threshold)
            x_partition_ends = 0:((x_length)/num_partitions):x_length;
            
            partition_entropies = zeros(num_partitions,1);
            valid_partitions = 0;
            
            
            for i = 1: num_partitions-1
                
                
                xpartbegin = x_partition_ends(i)+1;
                xpartend = x_partition_ends(i+1);
                
                
                
                DREVI_submatrix = DREVI(xpartbegin:xpartend, :);
                
                [part_entropy, valid_bit] = compute_sample_entropy_2d(DREVI_submatrix, num_partitions, prob_threshold);
                partition_entropies(i) = part_entropy;
                valid_partitions = valid_partitions+valid_bit;
                
            end
            
            
            avg_partition_entropies = sum(partition_entropies)*(1/valid_partitions);
            DREMI = total_entropy - avg_partition_entropies
            
            
        end
        
        
        
        
        function [R ] = corrcoef_edge(obj, channel1_name, channel2_name)
            
            %[points_x, points_y] = obj.pairwise_visualize(channel1_name, channel2_name,'no_plot');
            channel1 = obj.name_channel_map(channel1_name);
            channel2 = obj.name_channel_map(channel2_name);
            points_x = obj.data(:,channel1);
            points_y = obj.data(:,channel2);
            
            R = corrcoef(points_x, points_y);
        end
        
        
        
        
        
        function [ drevi, xaxis, yaxis, density, prob_normalized_density, cond_mean_x, cond_mean_y, dense_points_x, dense_points_y, point_weights] = compute_drevi(obj, channel1_name, channel2_name, varargin)
            
            channel1 = obj.name_channel_map(channel1_name)
            channel2 = obj.name_channel_map(channel2_name)
            X = obj.data(:,channel1);
            Y = obj.data(:,channel2);
            total_cells = size(obj.data,1);
            
            
            num_slices = 128;
            minxval = 0;
            minyval = 0;
            
            draw_plot = 1;
            avg_pts_threshold = .5;
            fix_limits = 0;
            maxyval = max(Y);
            maxxval = max(X);
            fixy = 0;
            show_marginals = 0;
            
            for i=1:length(varargin)-1
                
                
                if(strcmp(varargin{i},'Slices'))
                    num_slices = varargin{i+1};
                end
                
                if(strcmp(varargin{i},'MinMaxY'))
                    
                    minyval = varargin{i+1};
                    maxyval = varargin{i+2};
                    
                    fixy = 1;
                    
                end
                
                if(strcmp(varargin{i},'MinMaxX'))
                    
                    minxval = varargin{i+1};
                    maxxval = varargin{i+2};
                    fixy = 1;
                    
                end
                
                if(strcmp(varargin{i},'MinMaxFitX'))
                    
                    minxval = min(X);
                    
                end
                
                if(strcmp(varargin{i},'MinMaxFitY'))
                    
                    minyval = min(Y);
                    
                    
                end
                
                
                if(strcmp(varargin{i},'Minval'))
                    
                    minxval = varargin{i+1};
                    minyval = minxval;
                end
                if(strcmp(varargin{i},'limits'))
                    
                    fix_limits = 1;
                    limitvector = varargin{i+1};
                    minx = limitvector(1);
                    miny = limitvector(2);
                    maxx = limitvector(3);
                    maxy = limitvector(4);
                    
                    
                end
                if(strcmp(varargin{i},'avg_pts_threshold'))
                    
                    avg_pts = 0;
                    avg_pts_threshold = varargin{i+1};
                    
                end
                
                xlabel(channel1_name);
                ylabel(channel2_name);
                
                
            end
            
            
            
            for i=1:length(varargin)
                
                if(strcmp(varargin{i},'no_plot'))
                    draw_plot = 0;
                end
                if(strcmp(varargin{i},'show_marginals'))
                    show_marginals=1;
                end
            end
            
            if(fix_limits == 0)
                
                [bandwidth,density,Grid_X,Grid_Y]=kde2d([X Y],num_slices,[minxval minyval],[maxxval maxyval]);
                
            else
                
                [bandwidth,density,Grid_X,Grid_Y]=kde2d([X Y],num_slices,[minx miny],[maxx maxy]);
                
                
            end
            
            num_cols = size(density,2);
            num_rows = size(density,1);
            xaxis = Grid_X(1,:);
            yaxis = Grid_Y(:,1);
            
            normalized_density = zeros(num_rows,num_cols);
            prob_normalized_density = zeros(num_rows,num_cols);
            
            
            for i=1:num_cols
                
                
                normalized_density(:,i) = density(:,i)/max(density(:,i));
                prob_normalized_density(:,i) = density(:,i)/norm(density(:,i),1);
                
            end
            
            
            cond_mean_x = [];
            cond_mean_y = [];
            
            
            for i=1:num_cols
                
                
                
                max_indices = find(normalized_density(:,i)>= avg_pts_threshold);
                
                cond_mean_x = [cond_mean_x xaxis(i)];
                
                new_point_y = dot(yaxis(max_indices),normalized_density(max_indices,i))/sum(normalized_density(max_indices,i));
                if(isnan(new_point_y))
                    new_point_y = 0;
                end
                cond_mean_y = [cond_mean_y new_point_y];
                
                
                
            end
            
            dense_points_x=[];
            dense_points_y=[];
            point_weights = [];
            for i=1:num_cols
                
                
                max_indices = find(normalized_density(:,i)>= avg_pts_threshold);
                
                new_points = ones(1,length(max_indices)).*xaxis(i);
                new_point_weights = transpose(normalized_density(max_indices,i));
                new_point_weights = new_point_weights ./ (sum(new_point_weights));
                dense_points_x = [dense_points_x new_points];
                
                
                y_indices = max_indices;
                new_points_y = transpose(yaxis(y_indices));
                dense_points_y = [dense_points_y new_points_y];
                point_weights = [point_weights new_point_weights];
                
            end
            
            
            
            
            
            %now create the side bars
            
            
            colsum = sum(density,1);
            normalized_colsum = colsum./max(colsum);
            
            rowsum = sum(density,2);
            normalized_rowsum = rowsum./max(rowsum);
            
            blueval = 0;
            corner = ones(11,11).*blueval;
            
            yaxis_increment = .01;
            yaxis_top_bar = [];
            top_bar = [];
            zero_vector = zeros(1,length(normalized_colsum));
            for i=1:1
                top_bar = [top_bar; zero_vector];
                yaxis_top_bar = [yaxis_top_bar; max(yaxis)+(yaxis_increment*i)];
                
            end
            
            for i=1:10
                top_bar = [top_bar; normalized_colsum];
                yaxis_top_bar = [yaxis_top_bar; max(yaxis)+(yaxis_increment*i)];
            end
            
            
            xaxis_increment = .01;
            xaxis_side_bar = [];
            side_bar = [];
            zero_vector = zeros(length(normalized_rowsum),1);
            
            for i=1:1
                side_bar = [side_bar zero_vector];
                xaxis_side_bar = [xaxis_side_bar max(xaxis)+(xaxis_increment*i)];
                
            end
            
            for i=1:10
                side_bar = [side_bar normalized_rowsum];
                xaxis_side_bar = [xaxis_side_bar max(xaxis)+(xaxis_increment*i)];
            end
            
            
            matrix_to_plot = [normalized_density side_bar];
            top_bar = [top_bar corner];
            matrix_to_plot = [matrix_to_plot; top_bar];
            
            xaxis_to_plot = [xaxis xaxis_side_bar];
            yaxis_to_plot = [yaxis; yaxis_top_bar];
            
            
            
            
            if(draw_plot)
                figure;
                if(show_marginals==1)
                    imagesc(xaxis_to_plot,yaxis_to_plot, matrix_to_plot);
                else
                    imagesc(xaxis,yaxis, normalized_density);
                end
                set(gca,'YDir','normal');
                colormap(jet);
            end
            
            drevi = normalized_density;
            xlabel(channel1_name);
            ylabel(channel2_name);
            
        end
        
        
        
        
        
        function obj = cluster_gate_genes(obj, cluster_indices)
            
            new_data = obj.data(:,cluster_indices);
            obj.data = new_data;
            obj.genes = obj.genes(gated_indices);
            
            for i=1:length(obj.genes)
                
                obj.name_channel_map(obj.genes{i}) = i;
                
            end
        end
        
        
        function [obj, gated_indices] = threshold_gate_genes(obj, channel_name, thresh, greater)
            
            mean_vals = mean(obj.data)
            
            if(strcmp(greater,'gt'))
                gated_indices = find(mean_vals>thresh);
            end
            if(strcmp(greater,'lt'))
                gated_indices = find(mean_vals<thresh);
            end
            
            
            new_data = obj.data(:,gated_indices);
            obj.data = new_data;
            obj.genes = obj.genes(gated_indices);
            
            for i=1:length(obj.genes)
                
                obj.name_channel_map(obj.genes{i}) = i;
                
            end
            
        end
        
        function obj = impute_cells(obj, k)
            
            data_new = nan(size(obj.data,1)*k, size(obj.data,2));
            disp 'computing knn'
            idx = knnsearch(obj.data, obj.data, 'dist', 'correlation', 'k', k+1);
            % create new cell per edge by recombination or max or mean
            disp 'computing new cells'
            for I=1:size(idx,1)
                cell1 = obj.data(I,:);
                for J=2:k+1
                    cell2 = obj.data(idx(I,J),:);
                    S = randi(2,1,size(obj.data,2))-1;
                    cell_new = cell1 .* S + cell2 .* ~S;
                    data_new( ((I-1)*k) + (J-1) ,:) = cell_new;
                end
            end
            obj.data = data_new;
            
        end
        
        function [noise_DREMI] = compute_noise_dremi(obj, channel1_name, channel2_name)
            
            
            channel1 = obj.name_channel_map(channel1_name);
            channel2 = obj.name_channel_map(channel2_name);
            dataX = obj.data(:,channel1);
            dataY = obj.data(:,channel2);
            
            [shifted_normalized_density, xaxis, yaxis] = mean_shifted_density(obj, channel1_name, channel2_name);
            
            [noise_DREMI] = compute_dremi(obj,channel1_name, channel2_name,.6, 8,'drevi_matrix', shifted_normalized_density)
            
            
        end
        
        
        
        function [shifted_normalized_density, xaxis, yaxis] = mean_shifted_density(obj, channel1_name, channel2_name)
            
            
            [normalized_density, xaxis, yaxis, ~, ~, points_x, points_y] = compute_drevi(obj, channel1_name, channel2_name, 'no_plot');
            %outputs: [ drevi, xaxis, yaxis, density, prob_normalized_density, cond_mean_x, cond_mean_y, dense_points_x, dense_points_y, point_weights]
            
            
            noise_DREMI = 0;
            [rows, cols] = size(normalized_density);
            
            
            grid_step = abs(yaxis(2)-yaxis(1));
            max_shift = floor(max(points_y)/grid_step);
            %now just make a bigger matrix
            
            shifted_normalized_density = zeros(rows+max_shift,cols);
            
            for i=1:cols
                
                
                column_shift = floor(points_y(i)/grid_step);
                start_position = max_shift - column_shift+1;
                shifted_normalized_density(start_position:start_position+rows-1,i)=normalized_density(:,i);
            end
            
            additional_points = (1:max_shift).*grid_step;
            additional_points = min(yaxis) - additional_points;
            yaxis_old = yaxis;
            yaxis = [transpose(additional_points); yaxis];
            
            
            %                colormap(jet);
            %                imagesc(xaxis,yaxis, shifted_normalized_density);
            %                set(gca,'YDir','normal');
            
            
            
        end
        
        function [ residuals ] = plot_knn_regression(obj, channel1_name, channel2_name, f, varargin)
            
            channel1 = obj.name_channel_map(channel1_name);
            channel2 = obj.name_channel_map(channel2_name);
            X = obj.data(:,channel1);
            Y = obj.data(:,channel2);
            
            avg_pts_threshold = 0;
            smooth_param = 0.99;
            
            for i=1:length(varargin)-1
                if(strcmp(varargin{i},'avg_pts_threshold'))
                    avg_pts_threshold = varargin{i+1};
                end
                if(strcmp(varargin{i},'smooth_param'))
                    smooth_param = varargin{i+1};
                end
            end
            
            %colormap(jet);
            % sort on X
            [X, ind] = sort(X);
            Y = Y(ind);
            fsorted = f(ind);
            
            ind_sm = fsorted >= avg_pts_threshold;
            
            options = fitoptions('Method','smoothingspline','Weights', fsorted(ind_sm), 'SmoothingParam', smooth_param);
            Yfit = fit(X(ind_sm),Y(ind_sm),'smoothingspline',options);
            
            residuals = Y - Yfit(X);
            
            obj.scatter_plot2d_colored(channel1_name, channel2_name, 'colordata', f);
            hold on;
            plot(X(ind_sm), Yfit(X(ind_sm)), '-k', 'linewidth', 4);
        end
        
        function parmhat = fit_conditional_nb(obj, channel1_name, channel2_name, varargin)
            figure;
            plot_2d_scatter(obj, channel1_name, channel2_name);
            figure;
            [drevi, ~, yaxis, ~, ~, ~, cond_mean_y] = compute_drevi(obj, channel1_name, channel2_name, varargin{:});
            parmhat = nan(4,size(drevi,2));
            P = nan(size(drevi));
            for I=1:size(drevi,2)
                y_vec = drevi(:,I);
                y_vec = y_vec ./ sum(y_vec);
                y_vec(y_vec<0) = 0;
                mu = sum(y_vec .* yaxis);
                sigmasq = sum((yaxis - mu).^2 .* y_vec);
                parmhat(2,I) = mu; % mean
                parmhat(3,I) = sqrt(sigmasq)/mu; % CV
                parmhat(4,I) = sigmasq/mu; % fano
                P(:,I) = gampdf(yaxis, (mu.^2)./sigmasq, sigmasq/mu);
                P(:,I) = P(:,I) ./ max(P(:,I));
            end
            parmhat(1,:) = cond_mean_y;
            figure;
            subplot(4,1,1);
            imagesc(parmhat(1,:));
            title 'thresholded mean'
            subplot(4,1,2);
            imagesc(parmhat(2,:));
            title 'mean'
            subplot(4,1,3);
            imagesc(parmhat(3,:));
            title 'CV'
            subplot(4,1,4);
            imagesc(parmhat(4,:));
            title 'Fano'
            figure;
            imagesc(P);
            set(gca,'ydir','normal');
        end
        

        
        function sort_genes_by_corr(obj, v, out_file, varargin)
            M = obj.data;
            for i=1:length(varargin)-1
                if(strcmp(varargin{i},'data'))
                    M = varargin{i+1};
                end
            end
            R = corr(M, v(:));
            R(isnan(R)) = 0;
            [~, ind] = sort(R, 'descend');
            DS = dataset();
            DS.R = R(ind);
            DS.G = obj.genes(ind);
            export(DS,'File',out_file,'Delimiter','\t');
        end
        
        function plot_tsne_diffusion_colored(obj, d, varargin)
            npca = inf;
            perplexity = 20;
            log_z = false;
            ca = [0 100];
            n_eigs = 9;
            for i=1:length(varargin)-1
                if(strcmp(varargin{i},'npca'))
                    npca = varargin{i+1};
                end
                if(strcmp(varargin{i},'perplexity'))
                    perplexity = varargin{i+1};
                end
                if(strcmp(varargin{i},'logz'))
                    log_z = varargin{i+1}';
                end
                if(strcmp(varargin{i},'ca'))
                    ca = varargin{i+1}';
                end
                if(strcmp(varargin{i},'n_eigs'))
                    n_eigs = varargin{i+1}';
                end
            end
            UR = obj.diffusion_map(varargin{:}) + (rand/1e12);
            npca = min(size(UR,1), npca);
            mappedX = fast_tsne(UR(:,2:n_eigs+1), npca, perplexity, 0);
            figure;
            hold on;
            xlabel 't-sne 1'
            ylabel 't-sne 2'
            %point_densities = abs(V(:,end));
            if log_z
                d = log(abs(d));
            end
            scatter(mappedX(:,1), mappedX(:,2), 60, d, 'filled');
            colorbar;
            caxis(prctile(d,ca));
        end
        

        function scatter_plot2d_colored(obj, channel1_name, channel2_name, varargin)
            
            figure;
            
            channel1 = obj.name_channel_map(channel1_name);
            channel2 = obj.name_channel_map(channel2_name);
            color_by_data = false;
            color_data = [];
            specify_color = false; 
            color = [];
            
            for i=1:length(varargin)-1
               
                if(strcmp(varargin{i},'colordata'))
                    
                    color_by_data = true;
                    color_data = varargin{i+1};
                    
                end
                
                if(strcmp(varargin{i},'color'))
                    
                    specify_color = true;
                    color = varargin{i+1};
                    
                end
            end
            ms = 50;
            if(color_by_data)
                scatter(obj.data(:,channel1), obj.data(:,channel2),ms, color_data, 'filled');
                ca_cd = [0 99];
                caxis(prctile(color_data,ca_cd));
                colormap(jet);
                colorbar;
        
            else
                if(specify_color)
                    
                    scatter(obj.data(:,channel1), obj.data(:,channel2) ,ms, color, 'filled');
                    
                else
                    
                    scatter(obj.data(:,channel1), obj.data(:,channel2) ,ms, 'filled');
                    
                end
            end
                %ca = [.5 99.5];
                ca = [0 100];
                xl = prctile(obj.data(:,channel1),ca);
                xlim(xl);
                %ca = [.5 99.5];
                ca = [0 100];
                yl = prctile(obj.data(:,channel2),ca);
                ylim(yl);

                xlabel(channel1_name);
                ylabel(channel2_name);
                

        end
        
        function plot_tsne_colored(obj, d, varargin)
            npca = 100;
            perplexity = 20;
            M = obj.data';
            log_z = false;
            ca = [0 100];
            U = [];
            for i=1:length(varargin)-1
                
                if(strcmp(varargin{i},'npca'))
                    npca = varargin{i+1};
                end
                
                if(strcmp(varargin{i},'perplexity'))
                    perplexity = varargin{i+1};
                end
                
                if(strcmp(varargin{i},'data'))
                    M = varargin{i+1}';
                end
                
                if(strcmp(varargin{i},'logz'))
                    log_z = varargin{i+1}';
                end
                
                if(strcmp(varargin{i},'ca'))
                    ca = varargin{i+1}';
                end
                
                if(strcmp(varargin{i},'U'))
                    U = varargin{i+1}';
                end
            end
            if ~isempty(U)
                pc = U';
                npca = size(U,1);
            else
                pc = pca(M', npca);
            end
            %mappedX = tsne(pc, [], pc(:,1:2), perplexity);
            mappedX = fast_tsne(pc, npca, perplexity, 0);
            figure;
            hold on;
            xlabel 't-sne 1'
            ylabel 't-sne 2'
            %point_densities = abs(V(:,end));
            if log_z
                d = log(abs(d));
            end
            scatter(mappedX(:,1), mappedX(:,2), 60, d, 'filled');
            colorbar;
            caxis(prctile(d,ca));
        end
        
      
        function [obj, cells_sampled] = subsample(obj, new_lib_size, min_lib_size, ncells_sampled)
            M = obj.data;
            cell_lib_size = sum(obj.data,2);
            idx = find(cell_lib_size<min_lib_size);
            cell_lib_size(idx)=[];
            M(idx,:)=[];
            [num_cells, num_genes] = size(M);
            %cells_sampled = randsample(num_cells, ncells_sampled, true, cell_lib_size);
            cells_sampled = randsample(num_cells, ncells_sampled, true);
            data_new = zeros(ncells_sampled, num_genes);
            for i = 1:ncells_sampled
                i
                current_cell = cells_sampled(i);
                cell_vector = zeros(cell_lib_size(current_cell),1);
                current_index = 1;
                for j = 1:num_genes
                    if(M(current_cell,j)==0)
                        continue;
                    end
                    gene_nums = ones(M(current_cell,j),1).*j;
                    last_index = current_index+length(gene_nums)-1;
                    cell_vector(current_index:last_index) = gene_nums;
                    current_index = last_index+1;
                end
                y = randsample(cell_lib_size(current_cell), new_lib_size);
                genes_sampled = cell_vector(y);
                for k=1:length(genes_sampled)
                    data_new(i,genes_sampled(k)) = data_new(i,genes_sampled(k))+1;
                end
            end
            obj.data = data_new;
            
%             data_new = obj.data;
%             ngenes = size(obj.data,2);
%             % remove cells that with library size < lib_size
%             lib_size_vec = sum(obj.data,2);
%             disp(['removing ' num2str(sum(lib_size_vec < lib_size)) ' of ' num2str(size(obj.data,1)) ' cells before sampling']);
%             data_new = data_new(lib_size_vec >= lib_size,:);
%             % randomize order of cells
%             data_new = data_new(randsample(1:size(data_new,1),size(data_new,1)),:);
%             % subsample lib_size molecules uniformly per cell for ncells
%             data_sample = zeros(ncells, ngenes);
%             for I=1:ncells
%                 sample_ind = mod(I-1,size(data_new,1))+1; % index of cell to sample from
%                 lib_size_this = sum(data_new(sample_ind,:));
%                 molecule_inds = randsample(lib_size_this, lib_size);
%                 for J = 1:length(molecule_inds) % iterate over sampled molecules
%                     % get gene of molecule
%                     pos = find((cumsum(data_new(sample_ind,:)) - molecule_inds(J)) >= 0);
%                     data_sample(I,pos(1)) = data_sample(I,pos(1)) + 1;
%                 end
%             end
%             obj.data = data_sample;
        end
        
        function [D, R] = spectral_density(obj, k_knn, k_eig, k_density)
            %% knn adjacency matrix
            disp 'computing knn adjacency matrix'
            sigma = 0.1;
            W = SimGraph_NearestNeighbors(obj.data', k_knn, 1, sigma);
            %% eigenvectors of laplacian (takes k_eig smallest eigenvalue eigenvectors)
            disp 'computing eigenvectors of the laplacian'
            [~, U] = Spectra(W, k_eig, 3); % Jordan normalisation
            %% knn euclidean on eigenvector space
            disp 'computing knn of eigenvectors'
            %remove first column of that
            [~, dist] = knnsearch(U, U, 'dist', 'euclidean', 'k', k_density+1);
            %% density using k_eig dimensional sphere
            disp 'computing point desnities'
            R = dist(:,end); % radius
            %V = ((pi^(k_eig/2))/gamma((k_eig/2)+1)) .* R.^k_eig; % volume of k_eig-dimensional sphere
            %D = (k_density+1) ./ V; % point densities
            D = (k_density+1) ./ R; % point densities, debug
        end
        
        %does not work right now
        function U = eigenvector_density(obj, k_knn)
        
            % knn adjacency matrix
            disp 'computing knn adjacency matrix'
            sigma = 17;
            W = SimGraph_NearestNeighbors(obj.data', k_knn, 1, sigma);
            % eigenvectors of laplacian (takes k_eig smallest eigenvalue eigenvectors)
            disp 'computing eigenvectors of the laplacian'
            % calculate degree matrix
            degs = sum(W, 2);
            D = sparse(1:size(W, 1), 1:size(W, 2), degs);
            % compute unnormalized Laplacian
            L = D - W;
            % avoid dividing by zero
            degs(degs == 0) = eps;
            % calculate D^(-1/2)
            D = spdiags(1./(degs.^0.5), 0, size(D, 1), size(D, 2));
            % calculate normalized Laplacian
            L = D * L * D;
            [U, ~] = eigs(L, 100, eps);
            U = bsxfun(@rdivide, U, sqrt(sum(U.^2, 2)));
            U = U(:,1);
       
        end
        
        function [C, W] = spectral_clustering(obj, k_knn, k_eig, k_means)
            disp 'computing knn adjacency matrix'
            sigma = 100;
            W = SimGraph_NearestNeighbors(obj.data', k_knn, 1, sigma);
            disp 'computing spectral clustering'
            [C, ~, ~] = SpectralClustering(W, k_eig, k_means, 3);
        end
        
        function sigma = search_sigma_empirical(~, A, varargin)
            disp 'using external function'
            sigma = search_sigma_empirical(A, varargin);
        end
        
        function sigma = search_sigma_max_eigenvector_entropyl(obj, A, varargin)
            disp 'searching sigma using maximum entropy of the first eigenvector'
            draw_plot = false;
            for i=1:length(varargin)-1
                if(strcmp(varargin{i},'draw_plot'))
                    draw_plot = varargin{i+1};
                end
            end

            n = 30;
            min_sigma = 0.15;
            max_sigma = 0.25;
            %max_sigma = 2*std(A(:));

            pts = linspace(min_sigma, max_sigma, n);
            entropies = nan(length(pts), 1);
            for I=1:n
                I
                sigma = pts(I);
                try
                    f = obj.centrality_diffusion('distmat', A, 'sigma', sigma); %, 'use_eigs', true, 'k_eig', 2);
                    %f(f==0) = [];
                    entropies(I) = compute_entropy(f);
                end
                %entropies(I) = obj.compute_plug_in_entropy_estimate(f, 2)
            end  
            
            pts_ip = linspace(min_sigma, max_sigma, 1000);
            entropies_ip = interp1(pts, entropies, pts_ip, 'linear');
            entropies_ip = smooth(entropies_ip, 10, 'lowess');
            [~,idx] = max(entropies_ip);
            sigma = pts_ip(idx);
            if(draw_plot)
                figure;
                hold on;
                plot(pts_ip, entropies_ip, '-b');
                plot(pts, entropies, '.k');
                plot(sigma, entropies_ip(idx), '*r', 'markersize', 20);
                xlabel('sigma');
                ylabel('entropy');
                title(['sigma = ' num2str(sigma)]);
            end
        end
        
        function [sigma] = search_sigma(~, A)
            sigma0 = std(A(:));
            options = optimoptions('fmincon');
            % Set TolFun to 1e-3 and set MaxTime to 1000
            options = optimoptions(options, 'TolFun', 1e-3, 'TolX', 1e-4);
            % Set the Display option to 'iter'
            options.Display = 'iter';
            sigma = fmincon(@(sigma)compute_exponential_entropy(sigma, A), sigma0, [],[],[],[],sigma0/2,1000,[], options);  
        end
        
        function f = sim_matrix(obj, k_knn, n_steps, n_init, varargin)
            L = obj.compute_laplacian(k_knn, varargin{:});
            n_states = size(L,1);
            F = zeros(n_states, n_init);
            for I=1:n_init
                I
                %ind = randperm(n_states);
                ri = randi(n_states);
                ind = [ri 1:ri-1 ri+1:n_states];
                [~,S] = hmmgenerate(n_steps, L(ind,ind), ones(n_states,1));
                P = zeros(n_states,1);
                T = tabulate(S);
                P(T(:,1)) = T(:,2);
                F(ind,I) = P ./ sum(P);
            end
            f = mean(F,2);
        end
        
        function obj = prune_data(obj, percent_range_to_keep)
            tpg = sum(obj.data,1);
            p = prctile(tpg, percent_range_to_keep);
            ind_keep = tpg > p(1) & tpg < p(2);
            obj.data = obj.data(:,ind_keep);
            obj.genes = obj.genes(ind_keep);
        end
        
        function cf = centrality_diffusion_conditional_density(obj, k_knn, data, conditioning_data)
            f = obj.centrality_diffusion_density(k_knn, 'data', data);
            c = obj.centrality_diffusion_density(k_knn, 'data', conditioning_data);
            cf = f./c; 
        end
                
        function [f, idx, dist] = euclidean_diffusion_density(obj, varargin)
            k_eig = 9;
            k_density = 4;
            for i=1:length(varargin)-1
                if(strcmp(varargin{i},'k_eig'))
                    k_eig = varargin{i+1};
                end
                if(strcmp(varargin{i},'k_density'))
                    k_density = varargin{i+1};
                end
            end
            UR = obj.diffusion_map(varargin{:});
            use_weights = 0;
            for i=1:length(varargin)-1
                if(strcmp(varargin{i},'weights'))
                    W = varargin{i+1}';
                    use_weights = 1;
                end
            end
            [idx, dist] = knnsearch(UR(:,2:k_eig+1),UR(:,2:k_eig+1),'k',k_density+1);
            D = dist(:,end);
            V = ((pi^(k_eig/2))/gamma((k_eig/2)+1)) .* D.^k_eig;
            %f = k_knn ./ D;
            num_points = size(obj.data,1);
            if(use_weights == 0)
                %f = (k_density-1) ./ (V*num_points);
                %f = (k_density) ./ (dist(:,end).^k_eig);
                f = (k_density) ./ (dist(:,end));
            else
                neighbor_weight_sum = sum(W(idx),2);
                f = (neighbor_weight_sum) ./ (V*num_points);
            end
        end
        
        function f = euclidean_density(obj, varargin)
            
            M = obj.data;
            percent_replace = [];
            
            for i=1:length(varargin)-1
                if(strcmp(varargin{i},'data'))
                    M = varargin{i+1};
                end
                if(strcmp(varargin{i},'weights'))
                    W = varargin{i+1};
                end
                if(strcmp(varargin{i},'channels'))
                    channel_names = varargin{i+1};
                    channels = zeros(length(channel_names),1);
                    for j=1:length(channel_names)
                        channels(j) = obj.name_channel_map(channel_names{j});
                    end
                    M = obj.data(:,channels);
                end
                if(strcmp(varargin{i},'replace_low_density'))
                    percent_replace = varargin{i+1};
                end
            end
            
            k_knn = ceil(size(M,1)^(1/(1+size(M,2))));
            
            for i=1:length(varargin)-1
                if(strcmp(varargin{i},'k_density'))
                    k_knn = varargin{i+1};
                end
            end
            
            [~, dist] = knnsearch(M, M,'k',k_knn+1);
            num_dims = size(M,2);
            num_points = size(M,1);
            D = dist(:,end);
            V = ((pi^(num_dims/2))/gamma((num_dims/2)+1)) .* D.^num_dims;
            %f = k_knn ./ D;
            f = (k_knn-1) ./ (V*num_points);
            %f = (k_knn-1) ./ (V);
            if(~isempty(percent_replace))
                 p = prctile(f, percent_replace);
                 f(f<p) = p;
            end
            
%             [pf,pxi] = ksdensity(D);
%             figure;
%             plot(pxi,pf);
            
        end
        
        function Ak = low_rank_approximation_svd(~, A, k)
            disp 'in low_rank_approximation_svd'
            [U,S,V] = svd(full(A));
            S = diag(S);
            figure;
            plot(S, '.');
            S(k+1:end) = 0;
            S = diag(S);
            Ak = U * S * V';
        end
        
        function f = reduced_eigenvector_centrality_density(obj, varargin)
%             k_eig = 9;
%             for i=1:length(varargin)-1
%                 if(strcmp(varargin{i},'k_eig'))
%                     k_eig = varargin{i+1}';
%                 end
%             end
            [~,L,~] = compute_laplacian(obj, varargin{:});
            [~, EV, U] = eig(full(L));
            %U = bsxfun(@rdivide, U, sqrt(sum(U.^2, 2)));
            EV = diag(EV);
            [EV, ind] = sort(EV, 'descend');
            EV(1:10)
            U = U(:,ind);
            f = abs(U(:,1));
            
%             EV = abs(EV);
%             Un = U * sqrt(diag(EV));
%             EV(1:10)
%             L_k = Un(:,2:k_eig)*Un(:,2:k_eig)'; % reconstruct Laplacian
%             qq_norm = norm(L - L_k)
%             %L_k = Un(:,end-k_eig+1:end)*Un(:,end-k_eig+1:end)'; % reconstruct Laplacian
%             %L_k = -abs(L_k); % hack
%             %L_new(1:length(L_new)+1:end) = 1; % hack
%             [~, EV_k, U_k] = eig(L_k, 'balance');
%             %U_k = bsxfun(@rdivide, U_k, sqrt(sum(U_k.^2, 2)));
%             EV_k = diag(EV_k);
%             [EV_k, ind] = sort(EV_k, 'descend');
%             EV_k(1:10)
%             U_k = U_k(:,ind);
%             D_new = abs(U_k(:,1));
%             D_old = abs(U(:,1));
        end
        
        function A_new = diffusion_distance_matrix(obj, k_eig, sigma, varargin)
            % not working atm
            [~,~,W] = compute_laplacian(obj, varargin{:});
            W = full(W);
            [V, D] = eig(W);
            D_vec = diag(D);
            [~, sind] = sort(D_vec, 'ascend');
            %D_vec(sind([1 k_eig+2:end])) = 0;
            D_vec(sind(end:-1:(end-k_eig))) = 0;
            D = diag(D_vec);
            W_new = V*D*V'; % + repmat(mean(W),size(W,1),1);
            W_new(1:20)
            W(1:20)
            mean(W_new(:) - W(:))
            A_new = SimGraph_NearestNeighbors_similarity_reverse(W_new, sigma);
        end
        
        function [W,A] = spectral_partition(obj, k_knn, k_eig, k_cut, varargin)
            M = obj.data';
            draw_tsne = true;
            for i=1:length(varargin)-1
                if(strcmp(varargin{i},'data'))
                    M = varargin{i+1};
                end
                if(strcmp(varargin{i},'tsne'))
                    draw_tsne = varargin{i+1};
                end
            end
            disp(['k_eig = ' num2str(k_eig)]);
            disp 'computing knn distance matrix'
            A = SimGraph_NearestNeighbors_distance(M, k_knn, 1);
            disp 'searching sigma'
            sigma = obj.search_sigma_empirical(k_knn, A, 'draw_plot');
            disp(['sigma = ' num2str(sigma)]);
            disp 'computing knn weight matrix'
            W = SimGraph_NearestNeighbors_from_adjacency(A, k_knn, 1, sigma);
            % eigenvectors of laplacian (takes k_eig smallest eigenvalue eigenvectors)
            disp 'computing eigenvectors of the laplacian'
            % calculate degree matrix
            degs = sum(W, 2);
            D = sparse(1:size(W, 1), 1:size(W, 2), degs);
            % compute unnormalized Laplacian
            L = D - W;
            % avoid dividing by zero
            degs(degs == 0) = eps;
            % calculate D^(-1/2)
            D = spdiags(1./(degs.^0.5), 0, size(D, 1), size(D, 2));
            % calculate normalized Laplacian
            L = D * L * D;
            [U, ~] = eigs(L, k_eig, 'SM');
            U = bsxfun(@rdivide, U, sqrt(sum(U.^2, 2)));
            if draw_tsne
                % t-sne
                npca = 100;
                perplexity = 20;
                pc = pca(M', npca);
                mappedX = fast_tsne(pc, npca, perplexity, 0);
            else
                mappedX = M(1:2,:)';
            end
            % get partitions
            P = bi2de(U(:,2:k_cut+1) > 0);
            figure;
            hold on;
            C = unique(P);
            clr = hsv(length(C));
            %mt = {'x' 'o' 's' 'd' 'v' '^'};
            for I=1:length(C)
                ind = P == C(I);
                %plot(mappedX(ind,1), mappedX(ind,2), '.', 'color', clr(I,:), ...
                %    'displayname', num2str(C(I)), 'marker', mt{mod(I-1,length(mt))+1});
                plot(mappedX(ind,1), mappedX(ind,2), '.', 'color', clr(I,:), ...
                    'displayname', num2str(C(I)), 'markersize', 20);
            end
            legend('location','NW');
            if draw_tsne
                xlabel 't-sne 1'
                ylabel 't-sne 2'
            else
                xlabel 'x'
                ylabel 'y'
            end
        end
        
        function obj = impute_values_dd(obj, varargin)
            t = 0;
            L = [];
            for i=1:length(varargin)-1
                if(strcmp(varargin{i},'t'))
                    t = varargin{i+1};
                end
                if(strcmp(varargin{i},'laplacian'))
                    L = varargin{i+1};
                end
            end
            if isempty(L)
                [~,L] = obj.compute_laplacian(varargin{:});
            end
            obj.data = L^t * obj.data;
        end
        
%             [UR,UL,EV] = obj.diffusion_map(varargin{:});
%             if use_right
%                 U = UR;
%             else
%                 U = UL;
%             end
%             % ?t(x) = (?t1?1(x), ?t2?2(x), . . . , ?tk?k(x))
%             k_eig
%             %Phi_t = bsxfun(@times, U(:,2:k_eig+1), EV(2:k_eig+1)'.^t);
%             Phi_t = bsxfun(@times, U(:,2:k_eig+1), EV(2:k_eig+1)');
%             % D2(x0,x1) = ??(x0) ? ?(x1)?2
%             D = squareform(pdist(Phi_t));
%             %figure;
%             %hist(D(:),100);
%             %title(['\sigma = ' num2str(std(D(:))) ', \mu = ' num2str(mean(D(:)))]);
%             if strcmp(sigma_dd, 'auto')
%                 %sigma_dd = std(D(:)) / 2
%                 %sigma_dd = obj.search_sigma_empirical(D, 'draw_plot', true)
%                 [~, knn_dist] = knnsearch(Phi_t, Phi_t, 'k', size(obj.data,1)/20);
%                 sigma_dd = mean(knn_dist(:,end))
%             end
%             W = SimGraph_NearestNeighbors_similarity(D, sigma_dd);
%             degs = sum(W, 2);
%             D = sparse(1:size(W, 1), 1:size(W, 2), degs);
%             L = (D^-1) * W;
%             %L = L^t;
%             M = L * obj.data;
%             % only rescue zeros
%             if only_zero
%                 ind_zero = obj.data == 0;
%                 obj.data = obj.data + (ind_zero .* M);
%             else
%                 obj.data = M;
%             end
        
        function obj = impute_values_fpe(obj, varargin)
            k_eig = 9;
            only_zero = false;
            T = 1;
%             lib_size_vec = ones(size(obj.data,1),1);
            for i=1:length(varargin)-1
                if(strcmp(varargin{i},'k_eig'))
                    k_eig = varargin{i+1};
                end
                if(strcmp(varargin{i},'only_zero'))
                    only_zero = varargin{i+1};
                end
                if(strcmp(varargin{i},'T'))
                    T = varargin{i+1};
                end
%                 if(strcmp(varargin{i},'lib_size_vec'))
%                     lib_size_vec = varargin{i+1};
%                 end
            end
            % get left and right eigenvectors and eigenvalues
            [UR,UL,EV] = obj.diffusion_map(varargin{:});
            % compute new gene values per cell
            n_cells = size(obj.data,1);
            n_genes = size(obj.data,2);
            M = nan(size(obj.data));
            for x=1:n_cells % per cell (x)
                phi0_y = abs(UL(:,1)); % abs because can be flipped and negative!!!
                lambda = repmat(EV(2:k_eig+1)',n_cells,1);
                lambda = lambda.^T;
                psi_x = repmat(UR(x,2:k_eig+1),n_cells,1);
                phi_y = UL(:,2:k_eig+1);
                Q = lambda .* psi_x .* phi_y;
                P = phi0_y + sum(Q, 2);
                
                % debug:
                P = P.^5;
                %P(x) = 0.1;
                %P([1:x-1 x+1:end]) = (P([1:x-1 x+1:end]) ./ sum(P([1:x-1 x+1:end]))) .* (1-P(x));
                
                P = P ./ sum(P);
                
                %P(x)
                
%                 if any(P<0)
%                     x
%                     indneg = find(P<0)
%                     psi_x(indneg,:)
%                     phi_y(indneg,:)
%                     Q(indneg,:)
%                     P(indneg)
%                     error('negative')
%                 end
                P = repmat(P, [1 n_genes]);
                M(x,:) = sum(obj.data .* P);
            end
            % only rescue zeros
            if only_zero
                ind_zero = obj.data == 0;
                obj.data = obj.data + (ind_zero .* M);
            else
                obj.data = M;
            end
        end
        
    end
end
