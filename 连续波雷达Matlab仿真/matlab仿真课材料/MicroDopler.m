%% 直升机旋翼雷达微动多普勒特征测量程序
% 参考MATLAB程序编写
clear; close all; clc;

%% 参数设置
c = 3e8;                % 光速 (m/s)
fc = 10e9;              % 雷达载频 (Hz)
lambda = c/fc;          % 波长 (m)
PRF = 5000;             % 脉冲重复频率 (Hz)
T = 1;                  % 观测时间 (s)
N = T * PRF;            % 脉冲数
t = linspace(0, T, N);  % 时间序列

% 直升机参数
R0 = 1000;              % 初始距离 (m)
v = 50;                 % 直升机速度 (m/s)
theta = 30 * pi/180;    % 雷达视线与速度矢量夹角 (rad)

% 旋翼参数
L = 6;                  % 旋翼长度 (m)
omega_r = 4 * pi;       % 旋翼旋转角速度 (rad/s)
N_blades = 4;           % 叶片数量

%% 坐标系统定义
% 雷达位于原点，目标沿x轴运动
R_target = R0 + v * cos(theta) * t;  % 目标距离变化

%% 旋翼叶片微动建模
% 每个叶片上的散射点
N_points_per_blade = 10;
r_points = linspace(0.1, L, N_points_per_blade);  % 散射点位置
sigma_points = ones(1, N_points_per_blade);       % 散射点RCS

% 初始化微多普勒信号
s_micro = zeros(1, N);

%% 计算每个散射点的微多普勒
for blade_idx = 1:N_blades
    % 每个叶片的初始相位
    phi0 = 2*pi*(blade_idx-1)/N_blades;
    
    for point_idx = 1:N_points_per_blade
        r = r_points(point_idx);
        sigma = sigma_points(point_idx);
        
        % 叶片旋转引起的距离变化
        for n = 1:N
            % 叶片旋转角度
            phi = phi0 + omega_r * t(n);
            
            % 散射点在雷达视线方向的投影
            % 假设旋翼旋转平面垂直于直升机轴线
            delta_R = r * cos(phi) * sin(theta);
            
            % 总距离
            R_total = R_target(n) + delta_R;
            
            % 基带信号 (忽略幅度变化，关注相位)
            s_micro(n) = s_micro(n) + sigma * exp(-1j * 4*pi/lambda * R_total);
        end
    end
end

%% 信号处理 - 时频分析
% 短时傅里叶变换
window_length = 256;
overlap = window_length * 0.75;
nfft = 1024;

[s, f, t_spec] = spectrogram(s_micro, window_length, overlap, nfft, PRF, 'yaxis');

% 计算多普勒频率 (Hz)
f_doppler = f - 2*v*cos(theta)/lambda;

% 转换为速度 (m/s)
v_doppler = f_doppler * lambda / 2;

%% 绘图
figure('Position', [100, 100, 1200, 800]);

% 时频图
subplot(2,2,1);
imagesc(t_spec, v_doppler, 20*log10(abs(s)));
axis xy;
xlabel('时间 (s)');
ylabel('速度 (m/s)');
title('直升机旋翼微多普勒时频图');
colorbar;
grid on;

% 理论微多普勒曲线
subplot(2,2,2);
t_plot = linspace(0, T, 1000);
for blade_idx = 1:N_blades
    phi0 = 2*pi*(blade_idx-1)/N_blades;
    v_micro = -omega_r * L * sin(omega_r*t_plot + phi0) * sin(theta);
    plot(t_plot, v_micro, 'LineWidth', 1.5);
    hold on;
end
xlabel('时间 (s)');
ylabel('微多普勒速度 (m/s)');
title('理论微多普勒曲线');
legend('叶片1', '叶片2', '叶片3', '叶片4');
grid on;

% 距离-时间历史
subplot(2,2,3);
R_micro = zeros(size(t_plot));
for blade_idx = 1:N_blades
    phi0 = 2*pi*(blade_idx-1)/N_blades;
    R_blade = L * cos(omega_r*t_plot + phi0) * sin(theta);
    plot(t_plot, R_blade, 'LineWidth', 1.5);
    hold on;
end
xlabel('时间 (s)');
ylabel('距离变化 (m)');
title('旋翼叶片距离变化');
legend('叶片1', '叶片2', '叶片3', '叶片4');
grid on;

% 频谱分析
subplot(2,2,4);
N_fft = 4096;
f_axis = linspace(-PRF/2, PRF/2, N_fft);
S_fft = fftshift(fft(s_micro, N_fft));
v_axis = f_axis * lambda / 2;
plot(v_axis, 20*log10(abs(S_fft)));
xlabel('速度 (m/s)');
ylabel('幅度 (dB)');
title('多普勒频谱');
xlim([-50, 50]);
grid on;

%% 特征提取
fprintf('=== 直升机旋翼微多普勒特征分析 ===\n');
fprintf('旋翼长度: %.1f m\n', L);
fprintf('旋转频率: %.2f Hz\n', omega_r/(2*pi));
fprintf('叶片数量: %d\n', N_blades);
fprintf('最大微多普勒速度: ±%.2f m/s\n', omega_r * L * sin(theta));

% 调用参数估计函数
[est_omega, est_L, est_N] = estimate_microdoppler_params(s_micro, PRF, lambda, theta);

%% 微多普勒特征参数估计函数
function [est_omega, est_L, est_N] = estimate_microdoppler_params(s_micro, PRF, lambda, theta)
    % 简化的参数估计函数
    % 这里可以实现更复杂的参数估计算法
    
    % 使用频谱分析估计旋转频率
    N_fft = length(s_micro);
    f_axis = linspace(-PRF/2, PRF/2, N_fft);
    S_fft = fftshift(fft(s_micro, N_fft));
    
    % 寻找频谱峰值间隔来估计旋转频率
    [peaks, locs] = findpeaks(abs(S_fft), 'MinPeakHeight', max(abs(S_fft))*0.1);
    
    if length(locs) > 1
        peak_spacing = mean(diff(f_axis(locs)));
        est_omega = 2 * pi * abs(peak_spacing);
    else
        est_omega = 4 * pi;  % 默认值
    end
    
    % 估计旋翼长度 (简化方法)
    v_max = max(f_axis(locs)) * lambda / 2;
    est_L = v_max / (est_omega * sin(theta));
    
    % 估计叶片数量 (简化方法)
    est_N = round(2 * pi / (est_omega * 0.1));  % 基于周期性的简单估计
    
    fprintf('\n=== 参数估计结果 ===\n');
    fprintf('估计旋转频率: %.2f Hz\n', est_omega/(2*pi));
    fprintf('估计旋翼长度: %.2f m\n', est_L);
    fprintf('估计叶片数量: %d\n', est_N);
end