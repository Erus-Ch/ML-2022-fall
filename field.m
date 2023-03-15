%import data
filename = 'C:\Users\41227\Downloads\New-data\sss-_000001.dat';
flow_data = importdata(filename).data;
x_raw = flow_data(:,1);
y_raw = flow_data(:,2);
speed_raw = flow_data(:,7);

x = reshape(x_raw,[51,16]);
y = reshape(y_raw,[51,16]);
speed = reshape(speed_raw,[51,16]);

%plot speed
figure
contourf(x,y,speed,10)
contourf(x,y,speed,10,'ShowText','on')
xlabel('x/mm');ylabel('y/mm');
title('速度场(m/s)');

[cA1, cH1, cV1, cD1] = dwt2(speed, 'haar');
figure
subplot(221), imshow(cA1, []);
subplot(222), imshow(cH1, []);
subplot(223), imshow(cV1, []);
subplot(224), imshow(cD1, []);
