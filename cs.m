
%% 读取全部的原始数据
ori_data = zeros(816, 3380); % 816*3380
col = 1;
for id = 1:2:6759
name = ['C:\Users\41227\Downloads\New-data\sss-_',num2str(id,'%06d'),'.dat'];
imp_data = importdata(name).data(:,7);
ori_data(:,col) = imp_data;
col = col + 1;
end

%% 随机采样
% 从 3380 张照片中随机采样 500 张
sample = randperm(3380);
sample = sample(1:500);
sample = sort(sample);
sample_data = zeros(816, 1);
for i = 1:500
sample_data(:,i) = ori_data(:,sample(i));
end

% 构建采样矩阵
Phi = zeros(500, 3371);
for i = 1:500
Phi(i,sample(i)) = 1;
end

%% 对欠采样的数据进行 SVD 分解
[U,S,V] = svd(sample_data); % 查看分解结果，取前 2 阶
y1 = V(:,1); % 重构 V'矩阵的第一行，即重构 V 矩阵的第一列
y2 = V(:,2);

%% 压缩感知
n = 33;
Psi = inv(fft(eye(n, n)));
A = Phi * Psi;
cvx_begin
variable s1(n) complex;
minimize (norm(s1, 1));
subject to
A * s1 == y1;
cvx_end


%% 重构
x1 = real(ifft(s1));
x2 = real(ifft(s2));

rec_data = U(:,1:2) * S(1:2,1:2) * [x1,x2]';

for i=1:3380
vel_map = reshape(ori_data(:,i), 51, 16)';
img_name = ['ori_img/',num2str(i,'%06d'),'.jpg'];
imwrite(vel_map, img_name);
end
for i=1:3380
vel_map = reshape(rec_data(:,i), 51, 16)';
img_name = ['rec_img/',num2str(i,'%06d'),'.jpg'];
imwrite(vel_map, img_name);
end
