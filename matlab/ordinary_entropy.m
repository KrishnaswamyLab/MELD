function [entropy_val, valid_bit] = ordinary_entropy( y_distro )

valid_bit = 1;
y_distro_log=zeros(length(y_distro),1);
if (sum(y_distro)==0)
    entropy_val = 0;
    
    return;
end

y_distro_normalized = y_distro./sum(y_distro);
%y_distro = sum(data_matrix,2);
for i=1:length(y_distro)
    if((y_distro(i) == 0) || isnan(y_distro(i)))
        y_distro_log(i) = 0;
    else
        y_distro_log(i) = log2(y_distro_normalized(i));
    end
    
    entropy_val = -1*sum(y_distro_log.*y_distro_normalized);
    
end

