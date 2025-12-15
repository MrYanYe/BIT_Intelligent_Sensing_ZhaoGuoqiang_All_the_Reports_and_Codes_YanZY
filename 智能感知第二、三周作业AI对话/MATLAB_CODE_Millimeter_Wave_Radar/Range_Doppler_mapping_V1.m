clear; close all; clc;

%----------------------------
% 基本雷达与信号参数（可调整）
%----------------------------
c = 3e8;
fc = 77e9;            % 载波频率
lambda = c/fc;
B = 2e9;              % 带宽
Tc = 40e-6;           % 单次 chirp 时长（fast-time）
fs = 4e9;             % 采样率（fast-time）
S = B / Tc;           % 斜率
f0 = 0;               % 基带起始频率（0）
% 多普勒/慢时域（slow-time）设置
num_chirps = 128;     % 慢时间采样点（chirps 数）
PRI = 60e-6;          % 脉冲重复间隔（period between chirps），TC < PRI
% SNR（用于加噪声）
SNR_dB = 30;

%----------------------------
% 目标参数：两个目标
%----------------------------
R1_0 = 10; v1 = 5;    sigma1 = 0.8;
R2_0 = 15; v2 = -3;   sigma2 = 0.6;

%----------------------------
% 生成 fast-time 向量和 tx 基带 chirp（单个 chirp）
%----------------------------
dt = 1/fs;
t_fast = 0:dt:Tc-dt;              % fast-time samples per chirp
N_fast = length(t_fast);

% 预计算理想发射基带（单 chirp）
phi = 2*pi*( f0.*t_fast + 0.5 * S .* t_fast.^2 );
s_tx_chirp = exp(1j*phi);         % 1xN_fast

%----------------------------
% 逐 chirp 仿真：随慢时间更新目标距离，多普勒等
% 输出矩阵：RX_mat(dim: N_fast x num_chirps)
% 混频矩阵：BEAT_mat (N_fast x num_chirps)
%----------------------------
RX_mat = zeros(N_fast, num_chirps);
BEAT_mat = zeros(N_fast, num_chirps);

SNR_lin = 10^(SNR_dB/10);

for m = 1:num_chirps
    t0 = (m-1) * PRI;                         % slow-time 起点（s）
    % 当前 chirp 的全局时间向量（用于多普勒相位）
    t_global = t0 + t_fast;
    % 目标即时距离（R(t) = R0 + v * slow_time）
    R1 = R1_0 + v1 * t0;
    R2 = R2_0 + v2 * t0;
    % 往返时延
    tau1 = 2*R1 / c;
    tau2 = 2*R2 / c;
    % 衰减（工程近似）
    alpha1 = sigma1 * (1 / (R1^2));
    alpha2 = sigma2 * (1 / (R2^2));
    % 多普勒频移（基于一阶近似 fD = 2 v / lambda）
    fD1 = 2 * v1 / lambda;
    fD2 = 2 * v2 / lambda;

    % 生成回波：对 s_tx_chirp 做 fractional delay + 多普勒相位
    % 线性插值实现亚采样延迟
    delay_samples1 = tau1 * fs;
    delay_samples2 = tau2 * fs;
    s_rx1 = zeros(1,N_fast);
    s_rx2 = zeros(1,N_fast);
    n_idx = 0:N_fast-1;
    t_shift1 = n_idx - delay_samples1;
    t_shift2 = n_idx - delay_samples2;
    for k = 1:N_fast
        idx1 = t_shift1(k);
        if idx1 >= 1 && idx1 <= N_fast
            i0 = floor(idx1);
            frac = idx1 - i0;
            if i0 < 1
                val1 = 0;
            elseif i0+1 > N_fast
                val1 = s_tx_chirp(end);
            else
                val1 = (1-frac)*s_tx_chirp(i0) + frac*s_tx_chirp(i0+1);
            end
            % 多普勒以 (t - tau) 为参考相位
            ph1 = exp(1j*2*pi*fD1*( (k-1)/fs - tau1 ));
            s_rx1(k) = alpha1 * val1 .* ph1;
        end
        idx2 = t_shift2(k);
        if idx2 >= 1 && idx2 <= N_fast
            i0 = floor(idx2);
            frac = idx2 - i0;
            if i0 < 1
                val2 = 0;
            elseif i0+1 > N_fast
                val2 = s_tx_chirp(end);
            else
                val2 = (1-frac)*s_tx_chirp(i0) + frac*s_tx_chirp(i0+1);
            end
            ph2 = exp(1j*2*pi*fD2*( (k-1)/fs - tau2 ));
            s_rx2(k) = alpha2 * val2 .* ph2;
        end
    end

    % 合成回波并加入噪声（噪声依据回波功率设定）
    s_rx_total = s_rx1 + s_rx2;
    rx_power = mean(abs(s_rx_total).^2);
    if rx_power == 0
        rx_power = 1e-12;
    end
    noise_power_rx = rx_power / SNR_lin;
    noise_rx = sqrt(noise_power_rx/2) * (randn(size(s_rx_total)) + 1j*randn(size(s_rx_total)));
    s_rx_noisy = s_rx_total + noise_rx;

    % 发射信号也可以加噪（可选），此处保留理想发射或轻微噪声
    tx_noise = sqrt(mean(abs(s_tx_chirp).^2)/SNR_lin/2) * (randn(size(s_tx_chirp)) + 1j*randn(size(s_tx_chirp)));
    s_tx_noisy = s_tx_chirp + tx_noise;

    % 混频（基带）：tx * conj(rx)
    beat = s_tx_noisy .* conj(s_rx_noisy);

    % 存储
    RX_mat(:,m) = s_rx_noisy(:);
    BEAT_mat(:,m) = beat(:);
end

%----------------------------
% 2D FFT -> Range-Doppler 映射
% - fast-time 方向做 Range FFT（零填充提高分辨率）
% - slow-time 方向做 Doppler FFT（对 chirp 索引）
%----------------------------
% Range FFT 参数
Nfft_range = 4096;               % fast-time FFT length (range bins)
win_range = hann(N_fast);        % fast-time 窗
% Doppler FFT 参数
Nfft_doppler = 256;              % slow-time FFT length (velocity bins), 可 = num_chirps 或更大
win_doppler = hann(num_chirps); % slow-time 窗

% 应用窗函数
BEAT_win = (BEAT_mat .* repmat(win_range(:),1,num_chirps)) .* repmat(win_doppler(:).', N_fast, 1);

% 先对快时域做 FFT（沿列方向），得到 range profile per chirp
R_fft = fft(BEAT_win, Nfft_range, 1);    % dimension: Nfft_range x num_chirps

% 然后对慢时域（按每 range bin 做 doppler FFT）
RD_map = fftshift(fft(R_fft, Nfft_doppler, 2), 2); % shift slow-time (Doppler) to center

% 取幅度（dB）
RD_mag = 20*log10(abs(RD_map) + eps);

%----------------------------
% 频率/坐标换算：频率 -> 距离 / 速度
%----------------------------
% Range axis:
df_range = fs / Nfft_range;                    % fast-time 频率分辨
f_axis_range = (0:Nfft_range-1) * df_range;    % 双边但我们以 0..fs 看作基带
% 对 FMCW：beat 频率 fB < fs/2，对应距离 R = c * fB / (2 * S)
R_axis = (c * f_axis_range) / (2 * S);         % 单位 m

% Doppler axis:
fd_axis = (-Nfft_doppler/2 : Nfft_doppler/2-1) * (1/PRI) / Nfft_doppler * Nfft_doppler; % simplified
% 更直接：doppler bin spacing = 1 / (num_chirps * PRI) * Nfft_doppler ? 采用下面常规计算：
fd_bin = 1 / (PRI * Nfft_doppler);             % Hz per doppler bin
fd_axis = (-Nfft_doppler/2 : Nfft_doppler/2-1) * fd_bin;
% 速度轴（v = fd * lambda / 2）
v_axis = fd_axis * lambda / 2;

% 只取有效的 range 部分（fB 只能到 fs/2）
half_range = 1:floor(Nfft_range/2);
R_axis_use = R_axis(half_range);
RD_use = RD_mag(half_range, :);

%----------------------------
% 绘图：Range-Doppler 矩阵
%----------------------------
figure('Name','Range-Doppler Map','NumberTitle','off','Position',[200 200 900 600]);
imagesc(v_axis, R_axis_use, RD_use);
axis xy;
xlabel('Velocity (m/s)');
ylabel('Range (m)');
title('Range-Doppler Map (dB)');
colormap jet;
c = colorbar;
ylabel(c, 'Magnitude (dB)');

% 标注峰值（寻找最大峰坐标）
[~, idx_max] = max(RD_use(:));
[r_idx, v_idx] = ind2sub(size(RD_use), idx_max);
R_peak = R_axis_use(r_idx);
v_peak = v_axis(v_idx);
hold on;
plot(v_peak, R_peak, 'wo', 'MarkerSize',8, 'LineWidth',1.5);
text(v_peak, R_peak, sprintf('  (R=%.2f m, v=%.2f m/s)', R_peak, v_peak), 'Color','k','FontSize',10);
hold off;

%----------------------------
% 说明与注意
%----------------------------
fprintf('Peaks annotated: R_peak=%.3f m, v_peak=%.3f m/s\n', R_peak, v_peak);
