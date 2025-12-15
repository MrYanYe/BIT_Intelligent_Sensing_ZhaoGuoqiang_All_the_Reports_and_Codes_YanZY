clear; close all; clc;

% ----------------------------
% 公共参数（固定）
% ----------------------------
c = 3e8;
fc = 77e9;
lambda = c/fc;
Tc = 40e-6;           % chirp 时长
fs = 4e9;             % fast-time 采样率（足够高）
num_chirps = 128;     % slow-time chirps
PRI = 60e-6;
SNR_dB_rx = 40;       % 提高接收端SNR，保证回波明显（原来30dB可能太低）
SNR_lin = 10^(SNR_dB_rx/10);

% 目标（两个非常接近的静止目标，用以测量 range 分辨率）
R_center = 10;        % m
deltaR = 0.15;        % 目标间隔（设为理论 1 GHz 分辨率的量级）
R1 = R_center - deltaR/2;
R2 = R_center + deltaR/2;
v1 = 0; v2 = 0;
sigma1 = 1; sigma2 = 1;

% 待测带宽列表
B_list = [1e9, 2e9, 3e9];

% 绘图与 FFT 参数（统一）
Nfft_range = 16384;        % 大零填充，提升频谱细化（更接近连续谱）
Nfft_doppler = 256;
win_fast = hann(round(Tc*fs));    % fast-time 窗
win_slow = hann(num_chirps);      % slow-time 窗

% 只显示的 range 区间（聚焦在目标附近，便于比较）
R_display_min = 0;
R_display_max = 30;

% 结果存储
theo_DR = zeros(size(B_list));
meas_FWHM = zeros(size(B_list));

figure('Name','Range-Doppler for different B (improved)','NumberTitle','off','Position',[100 100 1400 420]);

for ib = 1:length(B_list)
    B = B_list(ib);
    S = B / Tc;
    theo_DR(ib) = c / (2*B);    % 理论距离分辨率

    % fast-time 向量与基带 chirp
    dt = 1/fs;
    t_fast = 0:dt:Tc-dt;
    N_fast = length(t_fast);
    phi = 2*pi*( 0.*t_fast + 0.5 * S .* t_fast.^2 );
    s_tx = exp(1j*phi);

    % 产生 BEAT_mat（N_fast x num_chirps）
    BEAT_mat = zeros(N_fast, num_chirps);
    for m = 1:num_chirps
        t0 = (m-1)*PRI;
        % 两个静止目标
        R1m = R1; R2m = R2;
        tau1 = 2*R1m / c; tau2 = 2*R2m / c;
        alpha1 = sigma1 / (R1m^2); alpha2 = sigma2 / (R2m^2);

        % fractional delay (linear interp) 高效实现：用 interp1（更稳健）
        t_samples = (0:N_fast-1)/fs;
        % 对发射信号延迟并缩放
        s_rx1 = alpha1 * interp1(t_samples, s_tx, t_samples - tau1, 'linear', 0);
        s_rx2 = alpha2 * interp1(t_samples, s_tx, t_samples - tau2, 'linear', 0);
        s_rx = s_rx1 + s_rx2;

        % 接收噪声按 rx_power 与目标 SNR 设置（保证回波在噪声上方）
        rx_power = mean(abs(s_rx).^2);
        if rx_power == 0, rx_power = 1e-12; end
        noise_rx = sqrt(rx_power / SNR_lin / 2) * (randn(size(s_rx)) + 1j*randn(size(s_rx)));
        s_rx_noisy = s_rx + noise_rx;

        % 发射信号可加小噪或不加
        s_tx_noisy = s_tx; 

        % 混频
        beat = s_tx_noisy .* conj(s_rx_noisy);

        % 去除每列直流（消除 slow-time 固定直通分量，减弱 v=0 竖线效应）
        beat = beat - mean(beat);

        BEAT_mat(:,m) = beat(:);
    end

    % 窗与 2D FFT（先 range，再 doppler）
    BEAT_win = (BEAT_mat .* repmat(win_fast(:),1,num_chirps)) .* repmat(win_slow(:).', N_fast, 1);
    R_fft = fft(BEAT_win, Nfft_range, 1);
    RD_map = fftshift(fft(R_fft, Nfft_doppler, 2), 2);
    RD_dB = 20*log10(abs(RD_map) + eps);

    % Range axis 转换
    df_range = fs / Nfft_range;
    f_axis = (0:Nfft_range-1) * df_range;
    R_axis = (c * f_axis) / (2 * S);
    half_idx = 1:floor(Nfft_range/2);
    R_axis_use = R_axis(half_idx);
    RD_use = RD_dB(half_idx, :);

    % 选取显示的 range 索引
    idx_r_min = find(R_axis_use >= R_display_min, 1, 'first');
    idx_r_max = find(R_axis_use <= R_display_max, 1, 'last');

    % 统一 color-scale：用所有图的最大值决定顶端以便可比
    if ib == 1
        global_peak = max(RD_use(:));
    else
        global_peak = max(global_peak, max(RD_use(:)));
    end

    % 暂存用于后续绘图（在循环外统一绘制 scale）
    RD_store{ib} = RD_use(idx_r_min:idx_r_max, :);
    R_axis_store{ib} = R_axis_use(idx_r_min:idx_r_max);
    v_axis = (-Nfft_doppler/2 : Nfft_doppler/2-1) * (1/(PRI*Nfft_doppler)); % doppler bin spacing
    v_axis = v_axis * lambda / 2;  % 转速度
    V_axis_store{ib} = v_axis;

    % 通过零多普勒列估计 FWHM（测量分辨率）
    dop_col = round(Nfft_doppler/2); % 零多普勒列索引
    range_profile = RD_use(:, dop_col);
    % 找在 R_center 附近的峰
    [pks, locs] = findpeaks(range_profile, 'SortStr','descend', 'NPeaks', 10);
    if isempty(locs)
        meas_FWHM(ib) = NaN;
    else
        % 选离 R_center 最近的峰（注意索引映射）
        R_vals_locs = R_axis_use(locs);
        [~, idx_closest] = min(abs(R_vals_locs - R_center));
        loc_peak = locs(idx_closest);
        peak_val = range_profile(loc_peak);
        half_power = peak_val - 3; % 3 dB 降幅
        % 左右交叉点
        left_idx = find(range_profile(1:loc_peak) <= half_power, 1, 'last');
        if isempty(left_idx), left_idx = 1; end
        right_rel = find(range_profile(loc_peak:end) <= half_power, 1, 'first');
        if isempty(right_rel), right_rel = length(range_profile)-loc_peak+1; end
        right_idx = loc_peak + right_rel - 1;
        fwhm_bins = right_idx - left_idx;
        bin_size_m = R_axis_use(2) - R_axis_use(1);
        meas_FWHM(ib) = fwhm_bins * bin_size_m;
    end
end

% 统一绘图（相同 color scale）
for ib = 1:length(B_list)
    subplot(1, length(B_list), ib);
    imagesc(V_axis_store{ib}, R_axis_store{ib}, RD_store{ib});
    axis xy;
    colormap jet;
    clim = [global_peak-40 global_peak]; % 显示峰值到峰值-40 dB
    caxis(clim);
    xlabel('Velocity (m/s)'); ylabel('Range (m)');
    title(sprintf('B = %.0f MHz, theory ΔR=%.3f m', B_list(ib)/1e6, theo_DR(ib)));
    colorbar;
    hold on;
    % 标注真实两个目标位置
    plot([-1 1], [R1 R1], 'w--', 'LineWidth', 1);
    plot([-1 1], [R2 R2], 'w--', 'LineWidth', 1);
    % 标注测量 FWHM
    text( max(V_axis_store{ib})*0.6, R_center+1, sprintf('meas FWHM=%.3f m', meas_FWHM(ib)), 'Color','w','FontSize',10 );
    hold off;
end

% 打印对比表
fprintf('B (GHz)   Theoretical ΔR (m)    Measured FWHM (m)\n');
for ib = 1:length(B_list)
    fprintf('%.3f       %.6f             %.6f\n', B_list(ib)/1e9, theo_DR(ib), meas_FWHM(ib));
end
