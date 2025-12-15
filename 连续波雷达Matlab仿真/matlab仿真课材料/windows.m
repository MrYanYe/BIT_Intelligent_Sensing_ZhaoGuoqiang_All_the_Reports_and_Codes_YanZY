% 窗函数教学演示程序
% 展示矩形窗、汉宁窗和切比雪夫窗的特性
% 包括副瓣抑制和主瓣展宽效果

clear; clc; close all;

%% 参数设置
N = 256;                % 窗函数长度
fs = 1000;              % 采样频率(Hz)
f1 = 100;               % 信号频率(Hz)
A = 1;                  % 信号幅度

% 切比雪夫窗参数
cheb_ripple = 60;       % 切比雪夫窗副瓣衰减(dB)

%% 生成信号
t = (0:N-1)/fs;         % 时间向量
signal = A * cos(2*pi*f1*t); % 单频信号

%% 生成窗函数
% 矩形窗
rect_win = rectwin(N)';
% 汉宁窗
hann_win = hann(N)';
% 切比雪夫窗
cheb_win = chebwin(N, cheb_ripple)';

% 归一化窗函数
rect_win = rect_win / sum(rect_win);
hann_win = hann_win / sum(hann_win);
cheb_win = cheb_win / sum(cheb_win);

%% 应用窗函数
signal_rect = signal .* rect_win;
signal_hann = signal .* hann_win;
signal_cheb = signal .* cheb_win;

%% 计算频谱
N_fft = 4096;  % FFT点数

% 矩形窗频谱
spec_rect = fft(signal_rect, N_fft);
spec_rect_db = 20*log10(abs(spec_rect)/max(abs(spec_rect)));

% 汉宁窗频谱
spec_hann = fft(signal_hann, N_fft);
spec_hann_db = 20*log10(abs(spec_hann)/max(abs(spec_hann)));

% 切比雪夫窗频谱
spec_cheb = fft(signal_cheb, N_fft);
spec_cheb_db = 20*log10(abs(spec_cheb)/max(abs(spec_cheb)));

% 频率轴
f = (0:N_fft-1)*fs/N_fft;
f = f(1:N_fft/2);  % 只取正频率部分

% 频谱也只取正频率部分
spec_rect_db = spec_rect_db(1:N_fft/2);
spec_hann_db = spec_hann_db(1:N_fft/2);
spec_cheb_db = spec_cheb_db(1:N_fft/2);

%% 绘制窗函数
figure('Position', [100, 100, 1200, 800]);

subplot(2,3,1);
plot(t, rect_win, 'LineWidth', 2);
title('矩形窗');
xlabel('时间 (s)');
ylabel('幅度');
grid on;
ylim([0, 1.1*max(rect_win)]);

subplot(2,3,2);
plot(t, hann_win, 'LineWidth', 2);
title('汉宁窗');
xlabel('时间 (s)');
ylabel('幅度');
grid on;
ylim([0, 1.1*max(hann_win)]);

subplot(2,3,3);
plot(t, cheb_win, 'LineWidth', 2);
title(['切比雪夫窗 (副瓣衰减: ', num2str(cheb_ripple), 'dB)']);
xlabel('时间 (s)');
ylabel('幅度');
grid on;
ylim([0, 1.1*max(cheb_win)]);

%% 绘制加窗后的信号
subplot(2,3,4);
plot(t, signal_rect, 'LineWidth', 1.5);
title('矩形窗加窗信号');
xlabel('时间 (s)');
ylabel('幅度');
grid on;

subplot(2,3,5);
plot(t, signal_hann, 'LineWidth', 1.5);
title('汉宁窗加窗信号');
xlabel('时间 (s)');
ylabel('幅度');
grid on;

subplot(2,3,6);
plot(t, signal_cheb, 'LineWidth', 1.5);
title('切比雪夫窗加窗信号');
xlabel('时间 (s)');
ylabel('幅度');
grid on;

%% 绘制频谱对比
figure('Position', [100, 100, 1200, 800]);

% 矩形窗频谱
subplot(3,1,1);
plot(f, spec_rect_db, 'LineWidth', 1.5);
title('矩形窗频谱');
xlabel('频率 (Hz)');
ylabel('幅度 (dB)');
xlim([0, 300]);
ylim([-100, 0]);
grid on;

% 标记主瓣宽度和副瓣电平
hold on;
[main_lobe_rect, side_lobe_rect] = analyze_spectrum(spec_rect_db, f, f1);
plot([f1, f1], [-100, 0], 'r--', 'LineWidth', 1);
text(f1+5, -20, ['主瓣宽度: ', num2str(main_lobe_rect, '%.1f'), 'Hz'], 'Color', 'red');
text(200, -30, ['最高副瓣: ', num2str(side_lobe_rect, '%.1f'), 'dB'], 'Color', 'blue');

% 汉宁窗频谱
subplot(3,1,2);
plot(f, spec_hann_db, 'LineWidth', 1.5);
title('汉宁窗频谱');
xlabel('频率 (Hz)');
ylabel('幅度 (dB)');
xlim([0, 300]);
ylim([-100, 0]);
grid on;

% 标记主瓣宽度和副瓣电平
hold on;
[main_lobe_hann, side_lobe_hann] = analyze_spectrum(spec_hann_db, f, f1);
plot([f1, f1], [-100, 0], 'r--', 'LineWidth', 1);
text(f1+5, -20, ['主瓣宽度: ', num2str(main_lobe_hann, '%.1f'), 'Hz'], 'Color', 'red');
text(200, -60, ['最高副瓣: ', num2str(side_lobe_hann, '%.1f'), 'dB'], 'Color', 'blue');

% 切比雪夫窗频谱
subplot(3,1,3);
plot(f, spec_cheb_db, 'LineWidth', 1.5);
title(['切比雪夫窗频谱 (副瓣衰减: ', num2str(cheb_ripple), 'dB)']);
xlabel('频率 (Hz)');
ylabel('幅度 (dB)');
xlim([0, 300]);
ylim([-100, 0]);
grid on;

% 标记主瓣宽度和副瓣电平
hold on;
[main_lobe_cheb, side_lobe_cheb] = analyze_spectrum(spec_cheb_db, f, f1);
plot([f1, f1], [-100, 0], 'r--', 'LineWidth', 1);
text(f1+5, -20, ['主瓣宽度: ', num2str(main_lobe_cheb, '%.1f'), 'Hz'], 'Color', 'red');
text(200, -cheb_ripple+5, ['最高副瓣: ', num2str(side_lobe_cheb, '%.1f'), 'dB'], 'Color', 'blue');

%% 绘制频谱对比（叠加显示）
figure('Position', [100, 100, 1200, 600]);

plot(f, spec_rect_db, 'LineWidth', 1.5);
hold on;
plot(f, spec_hann_db, 'LineWidth', 1.5);
plot(f, spec_cheb_db, 'LineWidth', 1.5);
title('窗函数频谱对比');
xlabel('频率 (Hz)');
ylabel('幅度 (dB)');
xlim([0, 300]);
ylim([-100, 0]);
grid on;
legend('矩形窗', '汉宁窗', ['切比雪夫窗(', num2str(cheb_ripple), 'dB)']);

% 标记主瓣宽度和副瓣电平
text(f1+5, -15, ['矩形窗主瓣: ', num2str(main_lobe_rect, '%.1f'), 'Hz'], 'Color', 'blue');
text(f1+5, -25, ['汉宁窗主瓣: ', num2str(main_lobe_hann, '%.1f'), 'Hz'], 'Color', 'red');
text(f1+5, -35, ['切比雪夫窗主瓣: ', num2str(main_lobe_cheb, '%.1f'), 'Hz'], 'Color', 'green');

text(200, -20, ['矩形窗副瓣: ', num2str(side_lobe_rect, '%.1f'), 'dB'], 'Color', 'blue');
text(200, -40, ['汉宁窗副瓣: ', num2str(side_lobe_hann, '%.1f'), 'dB'], 'Color', 'red');
text(200, -60, ['切比雪夫窗副瓣: ', num2str(side_lobe_cheb, '%.1f'), 'dB'], 'Color', 'green');

%% 绘制主瓣宽度和副瓣电平对比图
figure('Position', [100, 100, 1200, 600]);

subplot(1,2,1);
bar([main_lobe_rect, main_lobe_hann, main_lobe_cheb]);
set(gca, 'XTickLabel', {'矩形窗', '汉宁窗', '切比雪夫窗'});
ylabel('主瓣宽度 (Hz)');
title('主瓣宽度比较');

subplot(1,2,2);
bar([side_lobe_rect, side_lobe_hann, side_lobe_cheb]);
set(gca, 'XTickLabel', {'矩形窗', '汉宁窗', '切比雪夫窗'});
ylabel('最高副瓣电平 (dB)');
title('副瓣抑制比较');

%% 分析两个频率成分的情况（频谱泄露演示）
f2 = 105;  % 第二个频率成分，接近f1
signal2 = A * cos(2*pi*f1*t) + 0.5*A * cos(2*pi*f2*t); % 双频信号

% 应用窗函数
signal2_rect = signal2 .* rect_win;
signal2_hann = signal2 .* hann_win;
signal2_cheb = signal2 .* cheb_win;

% 计算频谱
spec2_rect = fft(signal2_rect, N_fft);
spec2_rect_db = 20*log10(abs(spec2_rect)/max(abs(spec2_rect)));
spec2_rect_db = spec2_rect_db(1:N_fft/2);

spec2_hann = fft(signal2_hann, N_fft);
spec2_hann_db = 20*log10(abs(spec2_hann)/max(abs(spec2_hann)));
spec2_hann_db = spec2_hann_db(1:N_fft/2);

spec2_cheb = fft(signal2_cheb, N_fft);
spec2_cheb_db = 20*log10(abs(spec2_cheb)/max(abs(spec2_cheb)));
spec2_cheb_db = spec2_cheb_db(1:N_fft/2);

% 绘制双频信号的频谱
figure('Position', [100, 100, 1200, 800]);

subplot(3,1,1);
plot(f, spec2_rect_db, 'LineWidth', 1.5);
title('矩形窗 - 双频信号频谱');
xlabel('频率 (Hz)');
ylabel('幅度 (dB)');
xlim([80, 120]);
ylim([-80, 0]);
grid on;
hold on;
plot([f1, f1], [-80, 0], 'r--', 'LineWidth', 1);
plot([f2, f2], [-80, 0], 'r--', 'LineWidth', 1);

subplot(3,1,2);
plot(f, spec2_hann_db, 'LineWidth', 1.5);
title('汉宁窗 - 双频信号频谱');
xlabel('频率 (Hz)');
ylabel('幅度 (dB)');
xlim([80, 120]);
ylim([-80, 0]);
grid on;
hold on;
plot([f1, f1], [-80, 0], 'r--', 'LineWidth', 1);
plot([f2, f2], [-80, 0], 'r--', 'LineWidth', 1);

subplot(3,1,3);
plot(f, spec2_cheb_db, 'LineWidth', 1.5);
title(['切比雪夫窗(', num2str(cheb_ripple), 'dB) - 双频信号频谱']);
xlabel('频率 (Hz)');
ylabel('幅度 (dB)');
xlim([80, 120]);
ylim([-80, 0]);
grid on;
hold on;
plot([f1, f1], [-80, 0], 'r--', 'LineWidth', 1);
plot([f2, f2], [-80, 0], 'r--', 'LineWidth', 1);

%% 频谱分析函数
function [main_lobe_width, max_side_lobe] = analyze_spectrum(spec_db, f, f0)
    % 找到主瓣峰值位置
    [~, idx] = max(spec_db);
    f_peak = f(idx);
    
    % 计算主瓣宽度 (-3dB带宽)
    peak_power = spec_db(idx);
    half_power = peak_power - 3;
    
    % 找到主瓣左侧-3dB点
    left_idx = idx;
    while left_idx > 1 && spec_db(left_idx) > half_power
        left_idx = left_idx - 1;
    end
    f_left = interp1(spec_db(left_idx:left_idx+1), f(left_idx:left_idx+1), half_power);
    
    % 找到主瓣右侧-3dB点
    right_idx = idx;
    while right_idx < length(spec_db) && spec_db(right_idx) > half_power
        right_idx = right_idx + 1;
    end
    f_right = interp1(spec_db(right_idx-1:right_idx), f(right_idx-1:right_idx), half_power);
    
    % 计算主瓣宽度
    main_lobe_width = f_right - f_left;
    
    % 找到最高副瓣电平（排除主瓣附近区域）
    % 主瓣区域定义为峰值频率±2倍主瓣宽度
    main_lobe_region = (f > (f_peak - 2*main_lobe_width)) & (f < (f_peak + 2*main_lobe_width));
    side_lobes = spec_db;
    side_lobes(main_lobe_region) = -Inf;  % 排除主瓣区域
    
    max_side_lobe = max(side_lobes);
end