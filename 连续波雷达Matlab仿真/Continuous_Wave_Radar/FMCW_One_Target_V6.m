% 功能: FMCW雷达发射/回波/混频/距离FFT/速度FFT仿真（单目标），
%      将单次网格中心的 argmax 改为连续值估计并做统计（多次噪声试验）
% 说明: 保留原有所有图形，增加统计分析图（直方图、散点）和连续估计（质心法）
% 修改: 在图中的红点和红圈旁标注坐标（数值标签）
% 变更: 开头以数组形式给出多个 SNR，仿真对每个 SNR 做统计汇总，但只为数组的最后一个 SNR 绘制所有图形（其余 SNR 仅汇总打印）
%
clear;
close all;
clc;

%% 可调参数
SNR_dB_array = [10,  20,40,60];   % SNR 扫描数组（dB），在此处修改以控制多点扫描
Nmc = 200;           % Monte Carlo 次数，用于统计估计分布

% 将用于绘图和示例的 SNR 设为数组最后一个元素（只为最后一个 SNR 绘图）
SNR_dB_for_plots = SNR_dB_array(end);

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
B = c / (2 * rangeRes);                   % 发射信号带宽
Tchirp = 5.5 * 2 * maxR / c;              % 扫频时间
slope = B / Tchirp;                       % 调频斜率
endle_time = 6.3e-6;                      % 空闲时间

Nd = 128;                                 % chirp数量
Nr = 1024;                                % ADC采样点数
vres = (c / fc) / (2 * Nd * (Tchirp + endle_time));  % 速度分辨率 (m/s)
Fs = Nr / Tchirp;                         % 模拟信号采样频率

%% 信号时间轴与基带信号模板（无噪声）
t = linspace(0, Nd * Tchirp, Nr * Nd);    % 采样时间

% 初始化信号数组
Tx = zeros(1, length(t));                 
Rx = zeros(1, length(t));                 
Mix_clean = zeros(1, length(t));          % 无噪声的Mix，用于快速复制

% 目标距离和延迟时间（随时间变化）
r_t = target_range + target_velocity * t; 
td = 2 * r_t / c;

% 信号幅度调整（考虑RCS）
amplitude = 10^(target_rcs/20);           % 将dB转换为线性幅度

% 生成干净的发射、接收、中频信号（不含噪声）
for i = 1:length(t)
    Tx(i) = cos(2 * pi * (fc * t(i) + (slope * t(i)^2) / 2));
    Rx(i) = amplitude * cos(2 * pi * (fc * (t(i) - td(i)) + (slope * (t(i) - td(i))^2) / 2));
    Mix_clean(i) = Tx(i) .* Rx(i);
end

%% 预计算用于轴显示
freq = fc + slope * t(1:Nr);                 
freq_echo = fc + slope * (t(1:Nr) - td(1:Nr));
range_axis = (0:Nr/2-1) * c / (2 * B);       % 距离轴 (m)
doppler_axis = linspace(-Nd/2, Nd/2-1, Nd) * vres;  % 速度轴 (m/s)

%% ---------- 准备：为绘图用最后一个SNR生成示例含噪声Mix（其余SNR不绘图） ----------
signal_power_example = mean(Mix_clean.^2);
noise_power_example_plot = signal_power_example / (10^(SNR_dB_for_plots/10));
noise_example = sqrt(noise_power_example_plot) * randn(size(Mix_clean));
Mix_example = Mix_clean + noise_example;
%% ------------------------------------------------------------------------------

%% 原有可视化（保留所有图）  -- 注意：这些图使用 SNR_dB_for_plots 的示例 Mix_example
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

% 中频信号（加入噪声后第一个trial展示，使用最后一个SNR的示例Mix）
subplot(2,3,4);
plot(t(1:Nr)*1e6, Mix_example(1:Nr));
xlabel('时间 (\mus)');
ylabel('幅度');
title(sprintf('中频信号（混频后，含噪声示例，SNR=%d dB）', SNR_dB_for_plots));
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

%% 将中频信号重塑为矩阵: signal矩阵 (Nr x Nd)
signal_clean = reshape(Mix_clean, Nr, Nd);

% 中频信号时域图（3D）（保留）
figure;
mesh(1:Nd, (1:Nr)*c/(2*B), abs(signal_clean));
xlabel('Chirp索引');
ylabel('距离 (m)');
zlabel('幅度');
title('中频信号时域图（3D）');
colorbar;

% 距离维FFT（示例使用最后一个SNR的Mix_example）
signal_example = reshape(Mix_example, Nr, Nd);
sig_fft = fft(signal_example, Nr) ./ Nr;
sig_fft = abs(sig_fft);
sig_fft = sig_fft(1:(Nr/2), :);

figure;
plot(range_axis, sig_fft(:,1));
xlabel('距离 (m)');
ylabel('幅度');
title('第一个Chirp的距离FFT结果（示例）');
grid on;
hold on;
h_target_marker = plot([target_range], [max(sig_fft(:,1))*0.8], 'ro', 'MarkerSize', 8, 'LineWidth', 2);
legend('距离谱', '目标位置');

% 在红圈旁标注坐标（距离, 幅度）
x_off = 0.02 * (max(range_axis) - min(range_axis));
y_off = 0.05 * (max(sig_fft(:,1)) - min(sig_fft(:,1)));
txt = sprintf('%.2f m, %.3g', target_range, max(sig_fft(:,1))*0.8);
text(target_range + x_off, max(sig_fft(:,1))*0.8 + y_off, txt, 'Color', 'r', 'FontWeight', 'bold');

% 距离FFT结果三维图（保留）
figure;
mesh(1:Nd, range_axis, sig_fft);
xlabel('Chirp索引');
ylabel('距离 (m)');
zlabel('幅度');
title('距离维FFT结果（3D）');
colorbar;

%% 计算距离-多普勒矩阵（示例，使用最后一个SNR的Mix_example）
sig_fft2 = fft2(signal_example, Nr, Nd);
sig_fft2 = fftshift(sig_fft2, 2);
RDM = abs(sig_fft2);
RDM = RDM(1:Nr/2, :);
RDM_db = 10 * log10(RDM + eps);

figure;
mesh(doppler_axis, range_axis, RDM_db);
xlabel('速度 (m/s)');
ylabel('距离 (m)');
zlabel('幅度 (dB)');
title('距离多普勒谱（3D）- 示例');
colorbar;
hold on;
h3 = scatter3(target_velocity, target_range, max(RDM_db(:)), 100, 'ro', 'filled', 'LineWidth', 2);
legend('距离多普勒谱', '目标位置');

% 在3D图的红点旁标注坐标 (速度, 距离, dB)
z_val = max(RDM_db(:));
dx = 0.02 * (max(doppler_axis) - min(doppler_axis));
dy = 0.02 * (max(range_axis) - min(range_axis));
dz = 0.02 * (max(RDM_db(:)) - min(RDM_db(:)));
txt3 = sprintf('%.2f m/s, %.2f m, %.2f dB', target_velocity, target_range, z_val);
text(target_velocity + dx, target_range + dy, z_val + dz, txt3, 'Color', 'r', 'FontWeight', 'bold');

figure;
contourf(doppler_axis, range_axis, RDM_db, 20);
xlabel('速度 (m/s)');
ylabel('距离 (m)');
title('距离多普勒谱（2D等高线）- 示例');
colorbar;
hold on;
h_cont_target = plot(target_velocity, target_range, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
legend('距离多普勒谱', '目标位置');

% 在等高线图的红圈旁标注坐标
dx2 = 0.02 * (max(doppler_axis) - min(doppler_axis));
dy2 = 0.02 * (max(range_axis) - min(range_axis));
txt2 = sprintf('V=%.2f m/s\nR=%.2f m', target_velocity, target_range);
text(target_velocity + dx2, target_range + dy2, txt2, 'Color', 'r', 'FontWeight', 'bold');

%% 估计方法说明与实现（对每个 SNR 执行统计，但仅绘图最后一个 SNR）
% 目标: 不用单次argmax的网格中心作为测量值
% 方法: 对每次噪声试验在RDM上先找到粗略峰值，然后在其邻域使用加权质心（连续估计）
%       统计 Nmc 次试验得到估计分布，计算均值和标准差，并汇总不同 SNR 的结果

% 为SNR扫描预分配汇总变量
numSNR = length(SNR_dB_array);
summary_mean_range = zeros(numSNR,1);
summary_std_range = zeros(numSNR,1);
summary_mean_vel = zeros(numSNR,1);
summary_std_vel = zeros(numSNR,1);
summary_mean_peakSNRdB = zeros(numSNR,1);
summary_std_peakSNRdB = zeros(numSNR,1);

% 为最后一个 SNR 保存详细样本用于绘图
est_ranges_last = [];
est_vels_last = [];
est_peakSNRdB_last = [];

% 质心邻域参数（保持原样）
r_win = 6;    % 距离方向邻域半宽（bins）
d_win = 6;    % 多普勒方向邻域半宽（bins）

for iSNR = 1:numSNR
    SNR_dB = SNR_dB_array(iSNR);
    
    % 预分配用于统计（每个 SNR 的 Monte Carlo）
    est_ranges = zeros(Nmc,1);
    est_vels = zeros(Nmc,1);
    est_peakSNRdB = zeros(Nmc,1);

    % Monte Carlo 试验
    for mc = 1:Nmc
        % 为每次试验添加独立噪声到中频信号（保持SNR设定）
        signal_power = mean(Mix_clean.^2);
        noise_power = signal_power / (10^(SNR_dB/10));
        noise = sqrt(noise_power) * randn(size(Mix_clean));
        Mix_noisy = Mix_clean + noise;

        % 重塑并计算RDM
        signal_mat = reshape(Mix_noisy, Nr, Nd);
        S2 = fft2(signal_mat, Nr, Nd);
        S2 = fftshift(S2, 2);      % 在多普勒维中心化
        RDM_lin = abs(S2);
        RDM_lin_pos = RDM_lin(1:Nr/2, :);   % 只取正频率部分
        RDM_db_pos = 10*log10(RDM_lin_pos + eps);

        % 找到粗略峰值索引（在dB图上）
        [~, idx] = max(RDM_db_pos(:));
        [row_peak, col_peak] = ind2sub(size(RDM_db_pos), idx);

        % 构造邻域索引范围，注意边界
        r_min = max(1, row_peak - r_win);
        r_max = min(size(RDM_db_pos,1), row_peak + r_win);
        d_min = max(1, col_peak - d_win);
        d_max = min(size(RDM_db_pos,2), col_peak + d_win);

        % 取邻域的线性幅度作为权重（避免用dB做质心）
        local_patch = RDM_lin_pos(r_min:r_max, d_min:d_max);
        % 行/列索引网格（全局索引）
        [R_idx, D_idx] = ndgrid(r_min:r_max, d_min:d_max);
        weights = local_patch;
        wsum = sum(weights(:)) + eps;

        % 加权质心（行和列）
        r_centroid_idx = sum(R_idx(:) .* weights(:)) / wsum;
        d_centroid_idx = sum(D_idx(:) .* weights(:)) / wsum;

        % 将连续索引映射到物理量
        % 距离轴等距，直接按比例插值
        r_frac = r_centroid_idx - 1;  % matlab索引从1开始，对应range_axis(1) == 0
        est_range = interp1(0:(length(range_axis)-1), range_axis, r_frac, 'linear', 'extrap');

        % 多普勒列索引到速度映射
        est_vel = interp1(1:Nd, doppler_axis, d_centroid_idx, 'linear', 'extrap');

        % 记录估计
        est_ranges(mc) = est_range;
        est_vels(mc) = est_vel;

        % 记录峰值SNR (dB)：采用峰值点dB减去全局平均dB
        peak_val_db = RDM_db_pos(row_peak, col_peak);
        bg_db = mean(RDM_db_pos(:));
        est_peakSNRdB(mc) = peak_val_db - bg_db;
    end

    % 汇总该 SNR 的统计量
    summary_mean_range(iSNR) = mean(est_ranges);
    summary_std_range(iSNR) = std(est_ranges);
    summary_mean_vel(iSNR) = mean(est_vels);
    summary_std_vel(iSNR) = std(est_vels);
    summary_mean_peakSNRdB(iSNR) = mean(est_peakSNRdB);
    summary_std_peakSNRdB(iSNR) = std(est_peakSNRdB);

    % 如果这是最后一个 SNR，则保存详细样本以便绘图（保持变量名与原代码兼容）
    if iSNR == numSNR
        est_ranges_last = est_ranges;
        est_vels_last = est_vels;
        est_peakSNRdB_last = est_peakSNRdB;
        % 为了后续单次示例绘图，保留最后一次的 Mix_noisy（上一次循环结束时的值）
        Mix_noisy_last = Mix_noisy;  %#ok<NASGU>
    end
end

%% 汇总打印：SNR 扫描结果表格（规整输出）
fprintf('\n=== SNR 扫描统计汇总 ===\n');
fprintf(' SNR(dB) | mean_range(m) | std_range(m) | mean_vel(m/s) | std_vel(m/s) | mean_peakSNR(dB) | std_peakSNR(dB)\n');
fprintf('-----------------------------------------------------------------------------------------------\n');
for i = 1:numSNR
    fprintf('  %5.1f  |   %10.4f  |  %9.4f  |   %10.4f  |  %9.4f  |    %10.4f    |   %10.4f\n', ...
        SNR_dB_array(i), summary_mean_range(i), summary_std_range(i), ...
        summary_mean_vel(i), summary_std_vel(i), summary_mean_peakSNRdB(i), summary_std_peakSNRdB(i));
end

%% 打印最后一个 SNR（详表），速度分辨率与距离分辨率
last_idx = numSNR;
fprintf('\n=== 最后一个 SNR 详细测量结果 (SNR = %d dB, Nmc = %d) ===\n', SNR_dB_array(last_idx), Nmc);
fprintf('理论距离: %.2f m\n', target_range);
fprintf('估计距离均值: %.4f m, 标准差: %.4f m\n', summary_mean_range(last_idx), summary_std_range(last_idx));
fprintf('理论速度: %.2f m/s\n', target_velocity);
fprintf('估计速度均值: %.4f m/s, 标准差: %.4f m/s\n', summary_mean_vel(last_idx), summary_std_vel(last_idx));
fprintf('峰值SNR (dB) 均值: %.2f dB, 标准差: %.2f dB\n', summary_mean_peakSNRdB(last_idx), summary_std_peakSNRdB(last_idx));
fprintf('\n速度分辨率: %.6f m/s\n', vres);
fprintf('距离分辨率: %.1f m\n', rangeRes);

%% 绘制统计图（仅针对最后一个 SNR）
% 将最后一个 SNR 的样本赋值给原代码变量名以保持图形逻辑一致
est_ranges = est_ranges_last;
est_vels = est_vels_last;
est_peakSNRdB = est_peakSNRdB_last;
Mix_noisy = Mix_noisy_last;  % 用于后面示例RDM重绘

% 距离估计直方图
figure;
histogram(est_ranges, 'Normalization', 'pdf');
xlabel('估计距离 (m)');
ylabel('概率密度');
title(sprintf('距离估计分布 (SNR=%d dB, Nmc=%d)  均值=%.3fm  std=%.3fm', SNR_dB_for_plots, Nmc, summary_mean_range(last_idx), summary_std_range(last_idx)));
hold on;
h_xline = xline(target_range, 'r-', 'LineWidth', 2);
legend('估计分布', '真实距离');

% 在xline旁标注坐标
ax = gca;
x_off_hist = 0.02 * (ax.XLim(2) - ax.XLim(1));
y_off_hist = 0.05 * (ax.YLim(2) - ax.YLim(1));
text(target_range + x_off_hist, ax.YLim(2) - y_off_hist, sprintf('R=%.2f m', target_range), 'Color', 'r', 'FontWeight', 'bold');

% 速度估计直方图
figure;
histogram(est_vels, 'Normalization', 'pdf');
xlabel('估计速度 (m/s)');
ylabel('概率密度');
title(sprintf('速度估计分布 (SNR=%d dB, Nmc=%d)  均值=%.3fm/s  std=%.3fm/s', SNR_dB_for_plots, Nmc, summary_mean_vel(last_idx), summary_std_vel(last_idx)));
hold on;
xline(target_velocity, 'r-', 'LineWidth', 2);
legend('估计分布', '真实速度');

ax = gca;
x_off_hist = 0.02 * (ax.XLim(2) - ax.XLim(1));
y_off_hist = 0.05 * (ax.YLim(2) - ax.YLim(1));
text(target_velocity + x_off_hist, ax.YLim(2) - y_off_hist, sprintf('V=%.2f m/s', target_velocity), 'Color', 'r', 'FontWeight', 'bold');

% 距离-速度散点图（表示每次试验的估计）
figure;
scatter(est_vels, est_ranges, 25, est_peakSNRdB, 'filled');
colormap jet;
colorbar;
xlabel('估计速度 (m/s)');
ylabel('估计距离 (m)');
title(sprintf('每次试验的距离-速度估计散点 (SNR=%d dB, 点色表示峰值SNR dB)', SNR_dB_for_plots));
hold on;
h_scatter_target = plot(target_velocity, target_range, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
xlim([min([est_vels; target_velocity]) - 2, max([est_vels; target_velocity]) + 2]);
ylim([min([est_ranges; target_range]) - 2, max([est_ranges; target_range]) + 2]);

% 在散点图红圈旁标注坐标
ax = gca;
x_off = 0.02 * (ax.XLim(2) - ax.XLim(1));
y_off = 0.02 * (ax.YLim(2) - ax.YLim(1));
text(target_velocity + x_off, target_range + y_off, sprintf('V=%.2f m/s\nR=%.2f m', target_velocity, target_range), 'Color', 'r', 'FontWeight', 'bold');

% 峰值SNR分布
figure;
histogram(est_peakSNRdB, 'Normalization', 'pdf');
xlabel('峰值SNR (dB)');
ylabel('概率密度');
title(sprintf('峰值SNR分布 (SNR=%d dB, Nmc=%d)  均值=%.2fdB  std=%.2fdB', SNR_dB_for_plots, Nmc, summary_mean_peakSNRdB(last_idx), summary_std_peakSNRdB(last_idx)));
grid on;

%% 单次示例：在示例RDM上绘制估计点与质心邻域（用于可视化单次处理）
% 选择最后一次试验的RDM展示邻域（重新计算最后一次的RDM用于显示）
signal_mat = reshape(Mix_noisy, Nr, Nd);
S2 = fft2(signal_mat, Nr, Nd);
S2 = fftshift(S2, 2);
RDM_lin = abs(S2);
RDM_lin_pos = RDM_lin(1:Nr/2, :);
RDM_db_pos = 10*log10(RDM_lin_pos + eps);

% 再次找到峰值并画出邻域
[~, idx] = max(RDM_db_pos(:));
[row_peak, col_peak] = ind2sub(size(RDM_db_pos), idx);
r_min = max(1, row_peak - r_win);
r_max = min(size(RDM_db_pos,1), row_peak + r_win);
d_min = max(1, col_peak - d_win);
d_max = min(size(RDM_db_pos,2), col_peak + d_win);

figure;
contourf(doppler_axis, range_axis, RDM_db_pos, 30);
xlabel('速度 (m/s)');
ylabel('距离 (m)');
title(sprintf('示例RDM及质心邻域展示 (SNR=%d dB)', SNR_dB_for_plots));
colorbar;
hold on;
% 标注粗峰点
plot(doppler_axis(col_peak), range_axis(row_peak), 'kx', 'LineWidth', 2, 'MarkerSize', 10);
% 标注邻域边界
plot(doppler_axis(d_min), range_axis(r_min), 'wo', 'MarkerSize', 6);
plot(doppler_axis(d_min), range_axis(r_max), 'wo', 'MarkerSize', 6);
plot(doppler_axis(d_max), range_axis(r_min), 'wo', 'MarkerSize', 6);
plot(doppler_axis(d_max), range_axis(r_max), 'wo', 'MarkerSize', 6);

% 计算实际质心位置用于显示
[R_idx, D_idx] = ndgrid(r_min:r_max, d_min:d_max);
weights = RDM_lin_pos(r_min:r_max, d_min:d_max);
wsum = sum(weights(:)) + eps;
r_centroid_idx = sum(R_idx(:) .* weights(:)) / wsum;
d_centroid_idx = sum(D_idx(:) .* weights(:)) / wsum;
r_centroid = interp1(0:(length(range_axis)-1), range_axis, r_centroid_idx-1, 'linear', 'extrap');
d_centroid = interp1(1:Nd, doppler_axis, d_centroid_idx, 'linear', 'extrap');
h_centroid = plot(d_centroid, r_centroid, 'r+', 'MarkerSize', 12, 'LineWidth', 2);
legend('RDM (dB)', '粗峰点', '邻域角点', '质心估计');

% 在示例RDM上为粗峰点和质心标注坐标
dx_rdm = 0.02 * (max(doppler_axis) - min(doppler_axis));
dy_rdm = 0.02 * (max(range_axis) - min(range_axis));
% 粗峰点坐标标签
txt_peak = sprintf('Peak: V=%.2f m/s\nR=%.2f m', doppler_axis(col_peak), range_axis(row_peak));
text(doppler_axis(col_peak) + dx_rdm, range_axis(row_peak) + dy_rdm, txt_peak, 'Color', 'k', 'FontWeight', 'bold');
% 质心坐标标签
txt_cent = sprintf('Centroid: V=%.3f m/s\nR=%.3f m', d_centroid, r_centroid);
text(d_centroid + dx_rdm, r_centroid - dy_rdm, txt_cent, 'Color', 'r', 'FontWeight', 'bold');

%% 结束输出
fprintf('\n=== FMCW雷达仿真完成（单目标版本，连续估计+统计）===\n');
fprintf('用于绘图的最后一个SNR: %d dB\n', SNR_dB_for_plots);
fprintf('速度分辨率: %.6f m/s\n', vres);
fprintf('距离分辨率: %.1f m\n', rangeRes);
