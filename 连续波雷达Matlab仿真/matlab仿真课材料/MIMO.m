%% FMCW雷达仿真与信号处理
% 清理工作区
clear; clc; close all;

%% 1. 雷达参数设置
c = physconst('LightSpeed');        % 光速
BW = 150e6;                         % 有效带宽(Hz)
fc = 77e9;                          % 载波频率(Hz)
numADC = 256;                       % 每个Chirp的ADC采样点数
numChirps = 256;                    % 每帧的Chirp数
numCPI = 10;                        % 帧数(CPI数)
T = 10e-6;                          % PRI(脉冲重复间隔)，默认不存在空闲时间
PRF = 1/T;                          % 脉冲重复频率
Fs = numADC/T;                      % 采样频率(Hz)
dt = 1/Fs;                          % 采样间隔(s)
slope = BW/T;                       % 调频斜率(Hz/s)
lambda = c/fc;                      % 波长(m)
N = numChirps * numADC * numCPI;    % 总ADC采样点数
t = linspace(0, T*numChirps*numCPI, N); % 时间轴
t_onePulse = 0:dt:dt*numADC-dt;     % 单个Chirp时间

numTX = 1;                          % 发射天线数
numRX = 8;                          % 接收天线数

Vmax = lambda/(T*4);               % 最大不模糊速度(m/s)
DFmax = 1/2*PRF;                    % 最大不模糊多普勒频率(Hz)
dR = c/(2*BW);                      % 距离分辨率(m)
Rmax = Fs*c/(2*slope);              % 最大不模糊距离(m)(TI文档)
Rmax2 = c/2/PRF;                    % 最大不模糊距离(m)(讲义)
dV = lambda/(2*numChirps*T);        % 速度分辨率(m/s)
d_rx = lambda/2;                    % 接收天线间距(m)
d_tx = 4*d_rx;                      % 发射天线间距(m)

N_Dopp = numChirps;                 % 多普勒FFT长度
N_range = numADC;                   % 距离FFT长度
N_azimuth = numTX*numRX;            % 方位维长度

R = 0:dR:Rmax-dR;                   % 距离轴
V = linspace(-Vmax, Vmax, numChirps); % 速度轴
ang_ax = -90:90;                    % 角度轴

%% 2. 目标参数设置
% 目标1参数
r1_radial = 50;                     % 目标1径向距离(m)
v1_radial = 10;                     % 目标1径向速度(m/s)
tar1_angle = -10;                   % 目标1角度(度)
r1_y = cosd(tar1_angle)*r1_radial;  % 目标1Y坐标
r1_x = sind(tar1_angle)*r1_radial;  % 目标1X坐标
v1_y = cosd(tar1_angle)*v1_radial;  % 目标1Y方向速度
v1_x = sind(tar1_angle)*v1_radial;  % 目标1X方向速度
r1 = [r1_x, r1_y, 0];               % 目标1位置向量

% 目标2参数
r2_radial = 100;                    % 目标2径向距离(m)
v2_radial = -15;                    % 目标2径向速度(m/s)
tar2_angle = 10;                    % 目标2角度(度)
r2_y = cosd(tar2_angle)*r2_radial;  % 目标2Y坐标
r2_x = sind(tar2_angle)*r2_radial;  % 目标2X坐标
v2_y = cosd(tar2_angle)*v2_radial;  % 目标2Y方向速度
v2_x = sind(tar2_angle)*v2_radial;  % 目标2X方向速度
r2 = [r2_x, r2_y, 0];               % 目标2位置向量

% 发射天线位置
tx_loc = cell(1, numTX);
figure;
for i = 1:numTX
    tx_loc{i} = [(i-1)*d_tx, 0, 0];
    scatter3(tx_loc{i}(1), tx_loc{i}(2), tx_loc{i}(3), 'b', 'filled');
    hold on;
end

% 接收天线位置
rx_loc = cell(1, numRX);
for i = 1:numRX
    rx_loc{i} = [tx_loc{numTX}(1) + d_tx + (i-1)*d_rx, 0, 0];
    scatter3(rx_loc{i}(1), rx_loc{i}(2), rx_loc{i}(3), 'r', 'filled');
end
title('天线布局');
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
legend('发射天线', '接收天线');
grid on;
hold off;

% 目标位置随时间变化
tar1_loc = zeros(length(t), 3);
tar1_loc(:, 1) = r1(1) + v1_x*t;
tar1_loc(:, 2) = r1(2) + v1_y*t;

tar2_loc = zeros(length(t), 3);
tar2_loc(:, 1) = r2(1) + v2_x*t;
tar2_loc(:, 2) = r2(2) + v2_y*t;

%% 3. 发射信号建模
delays_tar1 = cell(numTX, numRX);
delays_tar2 = cell(numTX, numRX);

for i = 1:numTX
    for j = 1:numRX
        % 计算目标1的延迟
        delays_tar1{i,j} = (vecnorm(tar1_loc - repmat(rx_loc{j}, length(t), 1), 2, 2) + ...
                           vecnorm(tar1_loc - repmat(tx_loc{i}, length(t), 1), 2, 2)) / c;
        
        % 计算目标2的延迟
        delays_tar2{i,j} = (vecnorm(tar2_loc - repmat(rx_loc{j}, length(t), 1), 2, 2) + ...
                           vecnorm(tar2_loc - repmat(tx_loc{i}, length(t), 1), 2, 2)) / c;
    end
end

%% 4. 接收信号模型
% 相位函数定义
phase = @(tx, fx) 2*pi*(fx.*tx + slope/2*tx.^2); % 发射信号相位

% 初始化混合信号
mixed = cell(numTX, numRX);

% 生成接收信号
for i = 1:numTX
    for j = 1:numRX
        fprintf('Processing Channel: %d/%d\n', j, numRX);
        
        % 初始化信号
        signal_t = zeros(1, N);
        signal_1 = zeros(1, N);
        signal_2 = zeros(1, N);
        
        for k = 1:numChirps*numCPI
            % 计算当前Chirp的索引范围
            idx_start = (k-1)*numADC + 1;
            idx_end = k*numADC;
            
            % 发射信号相位
            phase_t = phase(t_onePulse, fc);
            
            % 目标1的接收信号相位
            phase_1 = phase(t_onePulse - delays_tar1{i,j}(idx_start), fc);
            
            % 目标2的接收信号相位
            phase_2 = phase(t_onePulse - delays_tar2{i,j}(idx_start), fc);
            
            % 生成信号
            signal_t(idx_start:idx_end) = exp(1j*phase_t);
            signal_1(idx_start:idx_end) = exp(1j*(phase_t - phase_1));
            signal_2(idx_start:idx_end) = exp(1j*(phase_t - phase_2));
        end
        
        % 混合信号（两个目标的回波叠加）
        mixed{i,j} = signal_1 + signal_2;
    end
end

% 绘制局部信号
figure;
plot(t(1:1000), real(mixed{1,1}(1:1000)));
xlabel('时间 (s)');
ylabel('幅度');
title('接收信号（局部）');
grid on;

%% 5. 2D-FFT处理
% 重组雷达数据立方体
RDC = reshape(cat(3, mixed{:}), numADC, numChirps*numCPI, numRX*numTX);

% 初始化距离-多普勒图
RDMs = zeros(numADC, numChirps, numTX*numRX, numCPI);

% 对每个CPI进行2D-FFT
for i = 1:numCPI
    RD_frame = RDC(:, (i-1)*numChirps+1:i*numChirps, :);
    RDMs(:, :, :, i) = fftshift(fft2(RD_frame, N_range, N_Dopp), 2);
end

% 绘制第一个接收通道的第一个CPI的距离-多普勒图
figure;
imagesc(V, R, 20*log10(abs(RDMs(:, :, 1, 1)) / max(max(abs(RDMs(:, :, 1, 1))))));
colormap(jet(256));
clim_val = get(gca, 'clim');
caxis([clim_val(1)/2, 0]);
xlabel('速度 (m/s)');
ylabel('距离 (m)');
title('距离-多普勒图 (2D-FFT)');
colorbar;

% 绘制3D距离-多普勒图
figure;
mesh(V, R, 20*log10(abs(RDMs(:, :, 1, 1)) / max(max(abs(RDMs(:, :, 1, 1))))));
xlabel('速度 (m/s)');
ylabel('距离 (m)');
zlabel('归一化幅度 (dB)');
title('3D距离-多普勒图');
colorbar;

%% 6. CFAR检测
numGuard = 2;           % 保护单元数
numTrain = numGuard*2;  % 训练单元数
P_fa = 1e-5;            % 期望的虚警概率
SNR_OFFSET = -5;        % dB

% 转换为dB格式
RDM_dB = 10*log10(abs(RDMs(:, :, 1, 1)) / max(max(abs(RDMs(:, :, 1, 1)))));

% 执行CA-CFAR检测
[RDM_mask, cfar_ranges, cfar_dopps, K] = ca_cfar(RDM_dB, numGuard, numTrain, P_fa, SNR_OFFSET);

% 绘制CFAR结果
figure;
imagesc(V, R, RDM_mask);
xlabel('速度 (m/s)');
ylabel('距离 (m)');
title('CA-CFAR检测结果');
colorbar;

%% 7. 角度估计
%% 7.1 3D-FFT角度估计
rangeFFT = fft(RDC(:, 1:numChirps, :), N_range);
angleFFT = fftshift(fft(rangeFFT, length(ang_ax), 3), 3);
range_az = squeeze(sum(angleFFT, 2)); % 距离-方位图

% 绘制FFT距离-角度图
figure;
mesh(ang_ax, R, 20*log10(abs(range_az) ./ max(abs(range_az(:)))));
xlabel('方位角 (度)');
ylabel('距离 (m)');
zlabel('归一化幅度 (dB)');
title('FFT距离-角度图');
colorbar;

% 提取检测目标的角度信息
if K > 0
    doas = zeros(K, 181); % 波达方向
    figure;
    hold on; grid on;
    for i = 1:K
        doas(i, :) = fftshift(fft(rangeFFT(cfar_ranges(i), cfar_dopps(i), :), 181));
        plot(ang_ax, 10*log10(abs(doas(i, :))));
    end
    xlabel('方位角 (度)');
    ylabel('幅度 (dB)');
    title('检测目标的角度响应');
    legend(arrayfun(@(x) sprintf('目标%d', x), 1:K, 'UniformOutput', false));
    hold off;
end

%% 7.2 MUSIC算法角度估计
if K > 0
    d = 0.5; % 归一化天线间距
    M = numCPI; % 快拍数

    % 生成导向矢量
    a1 = zeros(numTX*numRX, length(ang_ax));
    for k = 1:length(ang_ax)
        a1(:, k) = exp(-1j * 2 * pi * d * (0:numTX*numRX-1)' * sind(ang_ax(k)));
    end

    % 初始化MUSIC谱
    music_spectrum = zeros(K, length(ang_ax));

    % 对每个检测到的目标执行MUSIC算法
    for i = 1:K
        Rxx = zeros(numTX*numRX, numTX*numRX);
        
        % 计算协方差矩阵
        for m = 1:M
            A = squeeze(RDMs(cfar_ranges(i), cfar_dopps(i), :, m));
            Rxx = Rxx + (1/M) * (A * A');
        end
        
        % 特征分解
        [Q, D] = eig(Rxx); % Q: 特征向量(列), D: 特征值
        [D, I] = sort(diag(D), 'descend');
        Q = Q(:, I); % 对特征向量排序，信号特征向量在前
        
        % 信号子空间和噪声子空间
        Qs = Q(:, 1); % 信号特征向量
        Qn = Q(:, 2:end); % 噪声特征向量
        
        % 计算MUSIC谱
        for k = 1:length(ang_ax)
            music_spectrum(i, k) = (a1(:, k)' * a1(:, k)) / (a1(:, k)' * (Qn * Qn') * a1(:, k));
        end
    end

    % 绘制MUSIC谱
    figure;
    hold on; grid on;
    for i = 1:K
        plot(ang_ax, 10*log10(abs(music_spectrum(i, :))));
    end
    xlabel('方位角 (度)');
    ylabel('MUSIC谱 (dB)');
    title('MUSIC算法角度估计');
    legend(arrayfun(@(x) sprintf('目标%d', x), 1:K, 'UniformOutput', false));
    hold off;

    %% 7.3 点云生成
    % 从MUSIC谱中提取角度估计
    if K >= 2
        [~, I1] = max(music_spectrum(1, :));
        angle1 = ang_ax(I1);
        [~, I2] = max(music_spectrum(2, :));
        angle2 = ang_ax(I2);
        
        % 转换为3D坐标
        coor1 = [cfar_ranges(1)*sind(angle1), cfar_ranges(1)*cosd(angle1), 0];
        coor2 = [cfar_ranges(2)*sind(angle2), cfar_ranges(2)*cosd(angle2), 0];
        
        % 绘制点云
        figure;
        hold on;
        title('目标3D坐标(点云)');
        scatter3(coor1(1), coor1(2), coor1(3), 100, 'm', 'filled', 'LineWidth', 9);
        scatter3(coor2(1), coor2(2), coor2(3), 100, 'b', 'filled', 'LineWidth', 9);
        xlabel('X (m)');
        ylabel('Y (m)');
        zlabel('Z (m)');
        legend('目标1', '目标2');
        grid on;
        hold off;
    end

    %% 7.4 MUSIC距离-角度谱
    rangeFFT = fft(RDC);
    range_az_music = zeros(N_range, length(ang_ax));

    for i = 1:N_range
        Rxx = zeros(numTX*numRX, numTX*numRX);
        
        % 计算协方差矩阵
        for m = 1:M
            A = squeeze(sum(rangeFFT(i, (m-1)*numChirps+1:m*numChirps, :), 2));
            Rxx = Rxx + (1/M) * (A * A');
        end
        
        % 特征分解
        [Q, D] = eig(Rxx);
        [D, I] = sort(diag(D), 'descend');
        Q = Q(:, I);
        
        % 信号子空间和噪声子空间
        Qs = Q(:, 1); % 信号特征向量
        Qn = Q(:, 2:end); % 噪声特征向量
        
        % 计算MUSIC谱
        for k = 1:length(ang_ax)
            music_spectrum2 = (a1(:, k)' * a1(:, k)) / (a1(:, k)' * (Qn * Qn') * a1(:, k));
            range_az_music(i, k) = music_spectrum2;
        end
    end

    % 绘制MUSIC距离-角度图
    figure;
    imagesc(ang_ax, R, 20*log10(abs(range_az_music) ./ max(abs(range_az_music(:)))));
    xlabel('方位角 (度)');
    ylabel('距离 (m)');
    title('MUSIC距离-角度图');
    colorbar;
end

%% CA-CFAR函数
function [RDM_mask, cfar_ranges, cfar_dopps, K] = ca_cfar(RDM_dB, numGuard, numTrain, P_fa, SNR_OFFSET)
    [numRange, numDoppler] = size(RDM_dB);
    RDM_mask = zeros(numRange, numDoppler);
    
    % 计算阈值因子
    alpha = numTrain * (P_fa^(-1/numTrain) - 1);
    
    cfar_ranges = [];
    cfar_dopps = [];
    K = 0;
    
    for rangeIdx = numGuard+numTrain+1 : numRange-numGuard-numTrain
        for dopplerIdx = numGuard+numTrain+1 : numDoppler-numGuard-numTrain
            % 获取CUT值
            cut = RDM_dB(rangeIdx, dopplerIdx);
            
            % 获取训练单元
            trainCells = [
                RDM_dB(rangeIdx-numGuard-numTrain : rangeIdx-numGuard-1, dopplerIdx); % 上方训练单元
                RDM_dB(rangeIdx+numGuard+1 : rangeIdx+numGuard+numTrain, dopplerIdx); % 下方训练单元
                RDM_dB(rangeIdx, dopplerIdx-numGuard-numTrain : dopplerIdx-numGuard-1)'; % 左侧训练单元
                RDM_dB(rangeIdx, dopplerIdx+numGuard+1 : dopplerIdx+numGuard+numTrain)' % 右侧训练单元
            ];
            
            % 计算阈值
            T = mean(trainCells) + alpha + SNR_OFFSET;
            
            % 检测
            if cut > T
                RDM_mask(rangeIdx, dopplerIdx) = 1;
                K = K + 1;
                cfar_ranges(K) = rangeIdx;
                cfar_dopps(K) = dopplerIdx;
            end
        end
    end
end