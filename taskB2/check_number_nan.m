function new_data = check_number_nan(data)

    new_data = zeros(size(data));
    for i=1:size(data,2)
        column = data(:,i);
        if (length(find(isnan(column))) < 0.9*length(column))
            new_data(:,i) = column;
        end   
    end
end