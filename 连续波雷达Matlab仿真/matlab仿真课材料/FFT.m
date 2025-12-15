% 离散傅里叶变换教学程序
% 展示时域和频域变换、频谱混叠、频谱泄露和频谱折叠

clear; clc; close all;

%% 基本参数设置
Fs = 1000;              % 采样频率 (Hz)
T = 1/Fs;               % 采样间隔 (s)
L = 1000;               % 信号长度
t = (0:L-1)*T;          % 时间向量

%% 1. 基本DFT变换 - 正弦波
f1 = 50;                % 频率 (Hz)
A1 = 1;                 % 振幅
x1 = A1*sin(2*pi*f1*t); % 正弦波信号

% 计算DFT
Y1 = fft(x1);
P2 = abs(Y1/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = Fs*(0:(L/2))/L;

% 绘制时域和频域图
figure('Position', [100, 100, 1200, 800]);
subplot(2,2,1);
plot(t, x1);
title('时域信号 - 正弦波');
xlabel('时间 (s)');
ylabel('幅度');
grid on;

subplot(2,2,2);
plot(f, P1);
title('频域表示 - 单频信号');
xlabel('频率 (Hz)');
ylabel('幅度');
xlim([0, Fs/2]);
grid on;

%% 2. 频谱混叠 - 高频信号
f2 = 750;               % 高于奈奎斯特频率的频率
x2 = A1*sin(2*pi*f2*t);

% 计算DFT
Y2 = fft(x2);
P2_2 = abs(Y2/L);
P2_1 = P2_2(1:L/2+1);
P2_1(2:end-1) = 2*P2_1(2:end-1);

subplot(2,2,3);
plot(t, x2);
title('时域信号 - 高频正弦波 (750Hz)');
xlabel('时间 (s)');
ylabel('幅度');
grid on;

subplot(2,2,4);
plot(f, P2_1);
title('频域表示 - 频谱混叠现象');
xlabel('频率 (Hz)');
ylabel('幅度');
xlim([0, Fs/2]);
grid on;

% 添加注释说明混叠
annotation('textbox', [0.75, 0.25, 0.15, 0.1], 'String', ...
    {'频谱混叠:', '750Hz信号在500Hz采样率下', '表现为250Hz的信号'}, ...
    'FitBoxToText', 'on', 'BackgroundColor', 'yellow');

%% 3. 频谱泄露 - 非整周期采样
figure('Position', [100, 100, 1200, 800]);

% 整周期采样
f3 = 50;                % 频率 (Hz)
cycles = 10;             % 周期数
L3 = round(cycles*(Fs/f3)); % 整周期采样点数
t3 = (0:L3-1)*T;
x3 = A1*sin(2*pi*f3*t3);

% 计算DFT
Y3 = fft(x3);
P3_2 = abs(Y3/L3);
P3_1 = P3_2(1:L3/2+1);
P3_1(2:end-1) = 2*P3_1(2:end-1);
f3_axis = Fs*(0:(L3/2))/L3;

subplot(2,2,1);
plot(t3, x3);
title('时域信号 - 整周期采样');
xlabel('时间 (s)');
ylabel('幅度');
grid on;

subplot(2,2,2);
plot(f3_axis, P3_1);
title('频域表示 - 无频谱泄露');
xlabel('频率 (Hz)');
ylabel('幅度');
xlim([0, Fs/2]);
grid on;

% 非整周期采样
L4 = L3 + 50;           % 非整周期采样点数
t4 = (0:L4-1)*T;
x4 = A1*sin(2*pi*f3*t4);

% 计算DFT
Y4 = fft(x4);
P4_2 = abs(Y4/L4);
P4_1 = P4_2(1:L4/2+1);
P4_1(2:end-1) = 2*P4_1(2:end-1);
f4_axis = Fs*(0:(L4/2))/L4;

subplot(2,2,3);
plot(t4, x4);
title('时域信号 - 非整周期采样');
xlabel('时间 (s)');
ylabel('幅度');
grid on;

subplot(2,2,4);
plot(f4_axis, P4_1);
title('频域表示 - 频谱泄露现象');
xlabel('频率 (Hz)');
ylabel('幅度');
xlim([0, Fs/2]);
grid on;

% 添加注释说明频谱泄露
annotation('textbox', [0.75, 0.25, 0.15, 0.1], 'String', ...
    {'频谱泄露:', '非整周期采样导致能量', '扩散到多个频率bin'}, ...
    'FitBoxToText', 'on', 'BackgroundColor', 'yellow');

%% 4. 频谱折叠 - 展示频率折叠现象
figure('Position', [100, 100, 1200, 800]);

% 生成多个频率成分的信号
f5 = [30, 80, 130, 300]; % 频率成分 (Hz)
A5 = [1, 0.8, 0.6, 0.4]; % 对应振幅
x5 = zeros(1, L);
for i = 1:length(f5)
    x5 = x5 + A5(i)*sin(2*pi*f5(i)*t);
end

% 计算DFT
Y5 = fft(x5);
P5_2 = abs(Y5/L);
P5_1 = P5_2(1:L/2+1);
P5_1(2:end-1) = 2*P5_1(2:end-1);

subplot(2,2,1);
plot(t, x5);
title('时域信号 - 多频成分');
xlabel('时间 (s)');
ylabel('幅度');
grid on;

subplot(2,2,2);
plot(f, P5_1);
title('频域表示 - 多频信号');
xlabel('频率 (Hz)');
ylabel('幅度');
xlim([0, Fs/2]);
grid on;

% 降低采样频率展示频谱折叠
Fs_low = 200;           % 降低采样频率
T_low = 1/Fs_low;
L_low = 500;            % 信号长度
t_low = (0:L_low-1)*T_low;

% 生成信号
x6 = zeros(1, L_low);
for i = 1:length(f5)
    x6 = x6 + A5(i)*sin(2*pi*f5(i)*t_low);
end

% 计算DFT
Y6 = fft(x6);
P6_2 = abs(Y6/L_low);
P6_1 = P6_2(1:L_low/2+1);
P6_1(2:end-1) = 2*P6_1(2:end-1);
f_low = Fs_low*(0:(L_low/2))/L_low;

subplot(2,2,3);
plot(t_low, x6);
title('时域信号 - 低采样率');
xlabel('时间 (s)');
ylabel('幅度');
grid on;

subplot(2,2,4);
plot(f_low, P6_1);
title('频域表示 - 频谱折叠现象');
xlabel('频率 (Hz)');
ylabel('幅度');
xlim([0, Fs_low/2]);
grid on;

% 添加注释说明频谱折叠
annotation('textbox', [0.75, 0.25, 0.15, 0.1], 'String', ...
    {'频谱折叠:', '300Hz信号在100Hz采样率下', '折叠为100Hz的信号'}, ...
    'FitBoxToText', 'on', 'BackgroundColor', 'yellow');

%% 5. 使用窗函数减少频谱泄露
figure('Position', [100, 100, 1200, 800]);

% 非整周期采样信号
f7 = 50;                % 频率 (Hz)
L7 = 1024;              % 信号长度
t7 = (0:L7-1)*T;
x7 = A1*sin(2*pi*f7*t7);

% 不加窗函数的DFT
Y7 = fft(x7);
P7_2 = abs(Y7/L7);
P7_1 = P7_2(1:L7/2+1);
P7_1(2:end-1) = 2*P7_1(2:end-1);
f7_axis = Fs*(0:(L7/2))/L7;

subplot(2,2,1);
plot(t7, x7);
title('时域信号 - 非整周期采样');
xlabel('时间 (s)');
ylabel('幅度');
grid on;

subplot(2,2,2);
plot(f7_axis, P7_1);
title('频域表示 - 无窗函数');
xlabel('频率 (Hz)');
ylabel('幅度');
xlim([0, Fs/2]);
grid on;

% 应用汉宁窗
win = hann(L7)';
x7_win = x7 .* win;

% 加窗后的DFT
Y7_win = fft(x7_win);
P7_win_2 = abs(Y7_win/L7);
P7_win_1 = P7_win_2(1:L7/2+1);
P7_win_1(2:end-1) = 2*P7_win_1(2:end-1);

subplot(2,2,3);
plot(t7, x7_win);
title('时域信号 - 应用汉宁窗');
xlabel('时间 (s)');
ylabel('幅度');
grid on;

subplot(2,2,4);
plot(f7_axis, P7_win_1);
title('频域表示 - 使用窗函数减少泄露');
xlabel('频率 (Hz)');
ylabel('幅度');
xlim([0, Fs/2]);
grid on;

% 添加注释说明窗函数的作用
annotation('textbox', [0.75, 0.25, 0.15, 0.1], 'String', ...
    {'窗函数作用:', '减少频谱泄露', '提高频率分辨率'}, ...
    'FitBoxToText', 'on', 'BackgroundColor', 'yellow');

%% 6. 综合示例：展示所有现象
figure('Position', [100, 100, 1200, 800]);

% 创建包含多个频率成分的信号，其中一些高于奈奎斯特频率
f_combined = [40, 120, 350, 600]; % 频率成分 (Hz)
A_combined = [1, 0.7, 0.5, 0.3]; % 对应振幅
x_combined = zeros(1, L);
for i = 1:length(f_combined)
    x_combined = x_combined + A_combined(i)*sin(2*pi*f_combined(i)*t);
end

% 计算DFT
Y_combined = fft(x_combined);
P_combined_2 = abs(Y_combined/L);
P_combined_1 = P_combined_2(1:L/2+1);
P_combined_1(2:end-1) = 2*P_combined_1(2:end-1);

subplot(2,1,1);
plot(t, x_combined);
title('时域信号 - 综合示例');
xlabel('时间 (s)');
ylabel('幅度');
grid on;

subplot(2,1,2);
plot(f, P_combined_1);
title('频域表示 - 综合示例 (包含混叠和折叠)');
xlabel('频率 (Hz)');
ylabel('幅度');
xlim([0, Fs/2]);
grid on;

% 添加注释说明
annotation('textbox', [0.75, 0.3, 0.2, 0.15], 'String', ...
    {'综合示例:', '40Hz: 正常频率', '120Hz: 正常频率', '350Hz: 折叠为150Hz', '600Hz: 混叠为400Hz'}, ...
    'FitBoxToText', 'on', 'BackgroundColor', 'yellow');

% 标记各个频率成分
hold on;
plot([40, 40], [0, max(P_combined_1)], 'r--');
plot([120, 120], [0, max(P_combined_1)], 'r--');
plot([150, 150], [0, max(P_combined_1)], 'g--'); % 350Hz折叠后的位置
plot([400, 400], [0, max(P_combined_1)], 'g--'); % 600Hz混叠后的位置
legend('频谱', '原始频率', '折叠/混叠频率');
hold off;

disp('离散傅里叶变换教学程序已完成');
disp('展示了以下现象:');
disp('1. 基本DFT变换');
disp('2. 频谱混叠 (Aliasing)');
disp('3. 频谱泄露 (Leakage)');
disp('4. 频谱折叠 (Folding)');
disp('5. 窗函数应用');