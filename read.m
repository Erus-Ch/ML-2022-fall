New_data_files = dir(['C:\Users\41227\Downloads\New-data\','sss*.dat']);
length_new_data_file = length(New_data_files);
Imp_data = importdata(New_data_files(1).name);
Sample_data = Imp_data.data;
Length_data = length(Sample_data);

Full_matrix = zeros(Length_data,length_new_data_file);
 
for i =1:length_new_data_file
    Imp_data = importdata(New_data_files(i).name);
    Instant_data = Imp_data.data;
    Instant_data = Instant_data(:,7);
    Full_matrix(:,i) = Instant_data;
end

ori_data = zeros(816, 3380); % 816*3380
col = 1;
for id = 1:2:6759
name = ['piv_data/sss-_',num2str(id,'%06d'),'.dat'];
imp_data = importdata(name).data(:,7);
ori_data(:,col) = imp_data;
col = col + 1;
end

sample = randperm(3380);
sample = sample(1:500);
sample = sort(sample);
sample_data = zeros(816, 0);
for i = 1:500
sample_data(:,i) = ori_data(:,sample(i));
end

[U,S,V] = svd(sample_data); 
y1 = V(:,1);
y2 = V(:,2);
