% 功能: FMCW雷达发射信号、回波信号、混频、距离维FFT、速度维FFT、CFAR检测建模仿真（多目标版本）

clear all;
close all;
clc;

%% 雷达系统参数设置
maxR = 200;           % 雷达最大探测距离(m)
rangeRes = 1;         % 雷达距离分辨率(m)
maxV = 70;            % 雷达最大检测速度(m/s)
fc = 77e9;            % 雷达工作频率载频(Hz)
c = 3e8;              % 光速(m/s)

%% 多目标参数设置（采用矩阵形式，方便扩展）
% 目标参数格式：[距离(m), 速度(m/s), RCS(dBsm)]
targets = [90,  20,  10;    % 目标1：距离90m, 速度20m/s, RCS=10dBsm
           100, 10,  8;     % 目标2：距离100m, 速度10m/s, RCS=8dBsm
           50,  -15, 12];   % 目标3：距离50m, 速度-15m/s, RCS=12dBsm

num_targets = size(targets, 1);  % 目标数量

fprintf('多目标参数设置：\n');
for i = 1:num_targets
    fprintf('目标%d: 距离=%.1fm, 速度=%.1fm/s, RCS=%.1fdBsm\n', ...
            i, targets(i,1), targets(i,2), targets(i,3));
end

%% FMCW波形参数设置
B = c / (2 * rangeRes);                   % 发射信号带宽 (B = 150MHz)
Tchirp = 5.5 * 2 * maxR / c;              % 扫频时间
slope = B / Tchirp;                       % 调频斜率
endle_time = 6.3e-6;                      % 空闲时间

Nd = 128;                                 % chirp数量
Nr = 1024;                                % ADC采样点数
vres = (c / fc) / (2 * Nd * (Tchirp + endle_time));  % 速度分辨率
Fs = Nr / Tchirp;                         % 模拟信号采样频率

%% 信号生成（多目标版本）
t = linspace(0, Nd * Tchirp, Nr * Nd);    % 采样时间

% 初始化信号数组
Tx = zeros(1, length(t));                 % 发射信号
Rx_total = zeros(1, length(t));           % 总接收信号（所有目标的叠加）
Mix = zeros(1, length(t));                % 中频信号

% 为每个目标创建独立的信号存储
Rx_targets = zeros(num_targets, length(t)); % 每个目标的接收信号
r_t_targets = zeros(num_targets, length(t)); % 每个目标的距离变化
td_targets = zeros(num_targets, length(t));  % 每个目标的延迟时间

freq = zeros(1, Nr);                      % 发射信号频率
freq_echo_targets = zeros(num_targets, Nr); % 每个目标的回波信号频率

% 多目标信号生成
for i = 1:length(t)
    % 发射信号（实数信号）
    Tx(i) = cos(2 * pi * (fc * t(i) + (slope * t(i)^2) / 2));
    
    % 为每个目标生成接收信号
    for k = 1:num_targets
        % 目标参数
        r0 = targets(k, 1);
        v0 = targets(k, 2);
        rcs = targets(k, 3);  % RCS用于信号幅度调整
        
        % 目标距离和延迟时间更新
        r_t_targets(k, i) = r0 + v0 * t(i);
        td_targets(k, i) = 2 * r_t_targets(k, i) / c;
        
        % 考虑RCS的信号幅度调整
        amplitude = 10^(rcs/20);  % 将dB转换为线性幅度
        
        % 单个目标的接收信号
        Rx_targets(k, i) = amplitude * cos(2 * pi * (fc * (t(i) - td_targets(k, i)) + ...
                          (slope * (t(i) - td_targets(k, i))^2) / 2));
    end
    
    % 总接收信号（所有目标的叠加）
    Rx_total(i) = sum(Rx_targets(:, i));
    
    % 时频分析（只取第一个chirp）
    if i <= Nr
        freq(i) = fc + slope * t(i);  % 发射信号频率
        
        for k = 1:num_targets
            freq_echo_targets(k, i) = fc + slope * (t(i) - td_targets(k, i));  % 回波信号频率
        end
    end
    
    % 混频得到中频信号
    Mix(i) = Tx(i) .* Rx_total(i);
end

%% 信号可视化
% 发射信号和接收信号对比图
figure('Position', [100, 100, 1200, 800]);

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

% 各目标接收信号
subplot(2,3,3);
colors = ['b', 'r', 'g', 'm', 'c'];  % 不同颜色用于不同目标
for k = 1:num_targets
    plot(t(1:Nr)*1e6, Rx_targets(k, 1:Nr), [colors(k) '-'], 'LineWidth', 1);
    hold on;
end
xlabel('时间 (\mus)');
ylabel('幅度');
title('各目标接收信号');
legend_str = arrayfun(@(k) sprintf('目标%d', k), 1:num_targets, 'UniformOutput', false);
legend(legend_str, 'Location', 'best');
grid on;

% 总接收信号
subplot(2,3,4);
plot(t(1:Nr)*1e6, Rx_total(1:Nr));
xlabel('时间 (\mus)');
ylabel('幅度');
title('总接收信号（多目标叠加）');
grid on;

% 各目标回波信号频率
subplot(2,3,5);
for k = 1:num_targets
    plot(t(1:Nr)*1e6, freq_echo_targets(k, 1:Nr)/1e9, [colors(k) '-'], 'LineWidth', 1.5);
    hold on;
end
xlabel('时间 (\mus)');
ylabel('频率 (GHz)');
title('各目标回波信号频率');
legend(legend_str, 'Location', 'best');
grid on;

% 中频信号
subplot(2,3,6);
plot(t(1:Nr)*1e6, Mix(1:Nr));
xlabel('时间 (\mus)');
ylabel('幅度');
title('中频信号（混频后）');
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
title('第一个Chirp的距离FFT结果（多目标）');
grid on;
hold on;

% 标记目标位置
for k = 1:num_targets
    plot([targets(k,1)], [max(sig_fft(:,1))*0.8], 'o', 'MarkerSize', 8, ...
         'LineWidth', 2, 'Color', colors(k));
end
legend('距离谱', legend_str{:});

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
title('距离多普勒谱（3D）- 多目标');
colorbar;
hold on;

% 标记目标位置 - 修复scatter3错误
for k = 1:num_targets
    % 使用正确的scatter3参数
    scatter3(targets(k,2), targets(k,1), max(RDM_db(:)), ...
             100, 'o', 'filled', 'MarkerEdgeColor', colors(k), ...
             'MarkerFaceColor', colors(k), 'LineWidth', 2);
end
legend_str_plot = ['距离多普勒谱', legend_str];
legend(legend_str_plot);

%% CFAR检测算法
% 参数设置
Tr = 8;           % 距离维参考单元数
Td = 4;           % 多普勒维参考单元数
Gr = 4;           % 距离维保护单元数
Gd = 2;           % 多普勒维保护单元数
snr_offset = 8;   % SNR偏移阈值(dB) - 提高阈值减少虚警

% 计算边界
r_margin = Tr + Gr;
d_margin = Td + Gd;

% 初始化CFAR结果矩阵
sig_CFAR = zeros(size(RDM_db));

% CFAR检测主循环
fprintf('开始CFAR检测...\n');
for i = (d_margin+1):(size(RDM_db,1)-d_margin)
    for j = (r_margin+1):(size(RDM_db,2)-r_margin)
        
        % 提取参考区域（包括保护单元）
        start_i = i - d_margin;
        end_i = i + d_margin;
        start_j = j - r_margin;
        end_j = j + r_margin;
        
        % 提取保护区域
        guard_start_i = i - Gd;
        guard_end_i = i + Gd;
        guard_start_j = j - Gr;
        guard_end_j = j + Gr;
        
        % 所有参考单元内的功率总和（排除保护单元）
        total_power = 0;
        ref_cell_count = 0;
        
        for m = start_i:end_i
            for n = start_j:end_j
                % 跳过保护单元
                if (m >= guard_start_i && m <= guard_end_i && ...
                    n >= guard_start_j && n <= guard_end_j)
                    continue;
                end
                % 确保索引在有效范围内
                if m >= 1 && m <= size(RDM_db,1) && n >= 1 && n <= size(RDM_db,2)
                    total_power = total_power + db2pow(RDM_db(m, n));
                    ref_cell_count = ref_cell_count + 1;
                end
            end
        end
        
        % 计算噪声电平和阈值
        if ref_cell_count > 0
            noise_level = pow2db(total_power / ref_cell_count);
            threshold = noise_level + snr_offset;
            
            % 将CUT与阈值进行比较
            if RDM_db(i, j) > threshold
                sig_CFAR(i, j) = 1;
            end
        end
    end
end

% 显示CFAR检测结果（3D）
figure;
[X, Y] = meshgrid(doppler_axis, range_axis);
mesh(X, Y, double(sig_CFAR));
xlabel('速度 (m/s)');
ylabel('距离 (m)');
zlabel('检测结果');
title('CFAR检测结果（3D）- 多目标');
zlim([0 1.5]);

%% 目标检测结果显示
[detect_i, detect_j] = find(sig_CFAR == 1);
if ~isempty(detect_i)
    fprintf('\n检测到 %d 个目标\n', length(detect_i));
    
    % 限制显示的检测点数量
    max_display_points = 20;
    if length(detect_i) > max_display_points
        fprintf('显示前%d个检测点（共%d个）\n', max_display_points, length(detect_i));
        detect_i = detect_i(1:max_display_points);
        detect_j = detect_j(1:max_display_points);
    end
    
    % 在距离多普勒谱上标记检测到的目标（3D）
    figure;
    mesh(doppler_axis, range_axis, RDM_db);
    hold on;
    
    % 标记检测到的目标 - 修复scatter3错误
    scatter3(doppler_axis(detect_j), range_axis(detect_i), ...
             max(RDM_db(:)) * ones(size(detect_i)), ...
             100, 'r', 'x', 'LineWidth', 3);
    
    % 标记理论目标位置 - 修复scatter3错误
    for k = 1:num_targets
        scatter3(targets(k,2), targets(k,1), max(RDM_db(:)), ...
                100, 'o', 'filled', 'MarkerEdgeColor', colors(k), ...
                'MarkerFaceColor', colors(k), 'LineWidth', 2);
    end
    
    xlabel('速度 (m/s)');
    ylabel('距离 (m)');
    zlabel('幅度 (dB)');
    title('距离多普勒谱与CFAR检测结果（3D）');
    legend_str_final = ['距离多普勒谱', '检测到的目标', legend_str];
    legend(legend_str_final);
    colorbar;
    
    % 显示检测到的目标信息
    fprintf('\n检测结果统计：\n');
    for k = 1:min(10, length(detect_i))
        range_detected = range_axis(detect_i(k));
        velocity_detected = doppler_axis(detect_j(k));
        fprintf('检测点%d: 距离=%.1fm, 速度=%.1fm/s\n', k, range_detected, velocity_detected);
    end
    
    % 检测性能评估
    fprintf('\n检测性能评估：\n');
    for k = 1:num_targets
        target_range = targets(k, 1);
        target_velocity = targets(k, 2);
        
        % 查找最近检测点
        range_errors = abs(range_axis(detect_i) - target_range);
        velocity_errors = abs(doppler_axis(detect_j) - target_velocity);
        total_errors = range_errors + velocity_errors;
        
        [min_error, min_idx] = min(total_errors);
        if min_error < 5  % 误差阈值
            fprintf('目标%d: 成功检测 (距离误差=%.2fm, 速度误差=%.2fm/s)\n', ...
                    k, range_errors(min_idx), velocity_errors(min_idx));
        else
            fprintf('目标%d: 可能漏检或误差较大\n', k);
        end
    end
else
    fprintf('未检测到目标\n');
end

%% 结果显示
fprintf('\n=== FMCW雷达仿真完成（多目标版本）===\n');
fprintf('目标数量: %d\n', num_targets);
fprintf('速度分辨率: %.3f m/s\n', vres);
fprintf('距离分辨率: %.1f m\n', rangeRes);
fprintf('CFAR检测阈值偏移: %.1f dB\n', snr_offset);

% 添加新目标的示例（取消注释以测试）
% targets = [targets; 120, -5, 9];  % 添加新目标
% num_targets = size(targets, 1);
% fprintf('新目标数量: %d\n', num_targets);