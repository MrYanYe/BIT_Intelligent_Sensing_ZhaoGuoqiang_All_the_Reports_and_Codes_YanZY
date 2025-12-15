
% 功能: FMCW雷达发射信号、回波信号、混频、距离维FFT、速度维FFT仿真（单目标版本）

clear all;
close all;
clc;

%% 雷达系统参数设置
maxR = 200;           % 雷达最大探测距离(m)
rangeRes = 1;         % 雷达距离分辨率(m)
maxV = 70;            % 雷达最大检测速度(m/s)
fc = 77e9;            % 雷达工作频率载频(Hz)
c = 3e8;              % 光速(m/s)

%% 单目标参数设置
target_range = 90;    % 目标距离(m)
target_velocity = 20; % 目标速度(m/s)
target_rcs = 10;      % 目标RCS(dBsm)

fprintf('单目标参数设置：\n');
fprintf('目标: 距离=%.1fm, 速度=%.1fm/s, RCS=%.1fdBsm\n', ...
        target_range, target_velocity, target_rcs);

%% FMCW波形参数设置
B = c / (2 * rangeRes);                   % 发射信号带宽 (B = 150MHz)
Tchirp = 5.5 * 2 * maxR / c;              % 扫频时间
slope = B / Tchirp;                       % 调频斜率
endle_time = 6.3e-6;                      % 空闲时间

Nd = 128;                                 % chirp数量
Nr = 1024;                                % ADC采样点数
vres = (c / fc) / (2 * Nd * (Tchirp + endle_time));  % 速度分辨率
Fs = Nr / Tchirp;                         % 模拟信号采样频率

%% 信号生成（单目标版本）
t = linspace(0, Nd * Tchirp, Nr * Nd);    % 采样时间

% 初始化信号数组
Tx = zeros(1, length(t));                 % 发射信号
Rx = zeros(1, length(t));                 % 接收信号
Mix = zeros(1, length(t));                % 中频信号

% 目标距离和延迟时间
r_t = target_range + target_velocity * t; % 目标距离随时间变化
td = 2 * r_t / c;                         % 延迟时间随时间变化

% 信号幅度调整（考虑RCS）
amplitude = 10^(target_rcs/20);            % 将dB转换为线性幅度

% 信号生成
for i = 1:length(t)
    % 发射信号（实数信号）
    Tx(i) = cos(2 * pi * (fc * t(i) + (slope * t(i)^2) / 2));
    
    % 接收信号
    Rx(i) = amplitude * cos(2 * pi * (fc * (t(i) - td(i)) + ...
                      (slope * (t(i) - td(i))^2) / 2));
    
    % 混频得到中频信号
    Mix(i) = Tx(i) .* Rx(i);
end

% 时频分析（只取第一个chirp）
freq = fc + slope * t(1:Nr);              % 发射信号频率
freq_echo = fc + slope * (t(1:Nr) - td(1:Nr)); % 回波信号频率

%% 信号可视化
% 发射信号和接收信号对比图
figure('Position', [100, 100, 1200, 600]);

% 发射信号时域图
subplot(2,3,1);
plot(t(1:Nr)*1e6, Tx(1:Nr));
xlabel('时间 (\mus)');
ylabel('幅度');
title('TX发射信号时域图');
grid on;

% 发射信号时频图
subplot(2,3,2);
plot(t(1:Nr)*1e6, freq(1:Nr)/1e9);
xlabel('时间 (\mus)');
ylabel('频率 (GHz)');
title('TX发射信号时频图');
grid on;

% 接收信号
subplot(2,3,3);
plot(t(1:Nr)*1e6, Rx(1:Nr));
xlabel('时间 (\mus)');
ylabel('幅度');
title('RX接收信号');
grid on;

% 中频信号
subplot(2,3,4);
plot(t(1:Nr)*1e6, Mix(1:Nr));
xlabel('时间 (\mus)');
ylabel('幅度');
title('中频信号（混频后）');
grid on;

% 发射和接收信号频率对比
subplot(2,3,5);
plot(t(1:Nr)*1e6, freq(1:Nr)/1e9, 'b-', 'LineWidth', 1.5);
hold on;
plot(t(1:Nr)*1e6, freq_echo(1:Nr)/1e9, 'r-', 'LineWidth', 1.5);
xlabel('时间 (\mus)');
ylabel('频率 (GHz)');
title('发射和接收信号频率对比');
legend('发射信号频率', '接收信号频率');
grid on;

% 中频信号频率（拍频）
subplot(2,3,6);
beat_freq = abs(freq - freq_echo);
plot(t(1:Nr)*1e6, beat_freq/1e6);
xlabel('时间 (\mus)');
ylabel('频率 (MHz)');
title('中频信号频率（拍频）');
grid on;

%% 信号重塑和距离维FFT
signal = reshape(Mix, Nr, Nd);

% 中频信号时域图（3D）
figure;
mesh(1:Nd, (1:Nr)*c/(2*B), abs(signal));
xlabel('Chirp索引');
ylabel('距离 (m)');
zlabel('幅度');
title('中频信号时域图（3D）');
colorbar;

% 距离维FFT
sig_fft = fft(signal, Nr) ./ Nr;
sig_fft = abs(sig_fft);
sig_fft = sig_fft(1:(Nr/2), :);

% 第一个chirp的FFT结果
figure;
range_axis = (0:Nr/2-1) * c / (2 * B);  % 距离轴
plot(range_axis, sig_fft(:,1));
xlabel('距离 (m)');
ylabel('幅度');
title('第一个Chirp的距离FFT结果');
grid on;
hold on;

% 标记目标位置
plot([target_range], [max(sig_fft(:,1))*0.8], 'ro', 'MarkerSize', 8, 'LineWidth', 2);
legend('距离谱', '目标位置');

% 距离FFT结果三维图
figure;
mesh(1:Nd, range_axis, sig_fft);
xlabel('Chirp索引');
ylabel('距离 (m)');
zlabel('幅度');
title('距离维FFT结果（3D）');
colorbar;

%% 速度维FFT（距离多普勒谱）
sig_fft2 = fft2(signal, Nr, Nd);
sig_fft2 = fftshift(sig_fft2, 2);  % 在多普勒维进行fftshift
RDM = abs(sig_fft2);
RDM = RDM(1:Nr/2, :);  % 只取正频率部分
RDM_db = 10 * log10(RDM + eps);  % 转换为dB尺度，避免log(0)

% 正确的轴生成
doppler_axis = linspace(-Nd/2, Nd/2-1, Nd) * vres;  % 速度轴
range_axis = (0:Nr/2-1) * c / (2 * B);  % 距离轴

% 距离多普勒谱（3D）
figure;
mesh(doppler_axis, range_axis, RDM_db);
xlabel('速度 (m/s)');
ylabel('距离 (m)');
zlabel('幅度 (dB)');
title('距离多普勒谱（3D）- 单目标');
colorbar;
hold on;

% 标记目标位置
scatter3(target_velocity, target_range, max(RDM_db(:)), ...
         100, 'ro', 'filled', 'LineWidth', 2);
legend('距离多普勒谱', '目标位置');

% 距离多普勒谱（2D等高线图）
figure;
contourf(doppler_axis, range_axis, RDM_db, 20);
xlabel('速度 (m/s)');
ylabel('距离 (m)');
title('距离多普勒谱（2D等高线）- 单目标');
colorbar;
hold on;

% 标记目标位置
plot(target_velocity, target_range, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
legend('距离多普勒谱', '目标位置');

%% 测量结果分析
% 在距离多普勒谱中找到峰值
[max_val, max_idx] = max(RDM_db(:));
[row, col] = ind2sub(size(RDM_db), max_idx);

% 计算测量结果
measured_range = range_axis(row);
measured_velocity = doppler_axis(col);

% 计算测量误差
range_error = abs(measured_range - target_range);
velocity_error = abs(measured_velocity - target_velocity);

% 显示测量结果
fprintf('\n=== 测量结果 ===\n');
fprintf('理论距离: %.2f m\n', target_range);
fprintf('测量距离: %.2f m\n', measured_range);
fprintf('距离误差: %.2f m\n', range_error);
fprintf('理论速度: %.2f m/s\n', target_velocity);
fprintf('测量速度: %.2f m/s\n', measured_velocity);
fprintf('速度误差: %.2f m/s\n', velocity_error);
fprintf('信噪比(峰值): %.2f dB\n', max_val - mean(RDM_db(:)));

%% 结果显示
fprintf('\n=== FMCW雷达仿真完成（单目标版本）===\n');
fprintf('速度分辨率: %.3f m/s\n', vres);
fprintf('距离分辨率: %.1f m\n', rangeRes);