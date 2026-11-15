function [tagco,y,ymod,maxobs] = readobs(filename)

%disp(filename);

fid = fopen(filename);

var = textscan(fid,'%d %d %f %f %f %f %f %f %f %f %f %f %f');

maxobs = numel(var{1});

tagco = double(cell2mat(var(:,7:13)));
y = double(cell2mat(var(:,5)));
ymod = double(cell2mat(var(:,6)));

fclose(fid);

end
    
