clear; close all; clc;

% ----------------------------
% 固定参数
% ----------------------------
c = 3e8;
fc = 77e9;
Tc = 40e-6;           % chirp 时长（固定）
fs = 4e9;             % fast-time 采样率（保持足够高以避免混淆）
SNR_dB = 30;
num_chirps = 128;     % slow-time 点数
PRI = 60e-6;
lambda = c/fc;

% 目标设置（两个紧邻目标用于测量分辨率）
R_target = 10;            % 基准距离 m
deltaR_true = 0.5;        % 两目标间隔初始值（会相对地可见/不可见）
R1 = R_target - deltaR_true/2;
R2 = R_target + deltaR_true/2;
v1 = 0; v2 = 0;           % 让两目标无多普勒以便只测 range 方向分辨
sigma1 = 1; sigma2 = 1;

% 待比较的带宽列表（Hz）
B_list = [1e9, 2e9, 3e9];

% 预分配保存测量结果
theo_DR = zeros(size(B_list));
meas_DR = zeros(size(B_list));   % 通过谱峰宽度估计的分辨率 (FWHM -> 换算成米)

% 可视化设置
figure('Name','Range-Doppler maps for different B','NumberTitle','off','Position',[100 100 1200 600]);

for ib = 1:length(B_list)
    B = B_list(ib);
    S = B / Tc;                     % 斜率
    % ----------------------------
    % 生成单 chirp 信号（fast-time）
    % ----------------------------
    dt = 1/fs;
    t_fast = 0:dt:Tc-dt;
    N_fast = length(t_fast);
    phi = 2*pi*( 0.*t_fast + 0.5 * S .* t_fast.^2 );
    s_tx_chirp = exp(1j*phi);
    % ----------------------------
    % 对每个 chirp 生成回波并混频（简化：两个静止目标）
    % ----------------------------
    BEAT_mat = zeros(N_fast, num_chirps);
    SNR_lin = 10^(SNR_dB/10);
    for m = 1:num_chirps
        t0 = (m-1)*PRI;
        % 目标瞬时距离（静止）
        R1m = R1; R2m = R2;
        tau1 = 2*R1m / c; tau2 = 2*R2m / c;
        alpha1 = sigma1 / (R1m^2); alpha2 = sigma2 / (R2m^2);
        % fractional delay (linear interp)
        delay1 = tau1 * fs; delay2 = tau2 * fs;
        s_rx1 = zeros(1,N_fast); s_rx2 = zeros(1,N_fast);
        idx = 0:N_fast-1;
        t_shift1 = idx - delay1; t_shift2 = idx - delay2;
        for k = 1:N_fast
            i1 = t_shift1(k);
            if i1 >= 1 && i1 <= N_fast
                i0 = floor(i1); frac = i1-i0;
                if i0 < 1
                    v1s = 0;
                elseif i0+1> N_fast
                    v1s = s_tx_chirp(end);
                else
                    v1s = (1-frac)*s_tx_chirp(i0) + frac*s_tx_chirp(i0+1);
                end
                s_rx1(k) = alpha1 * v1s;
            end
            i2 = t_shift2(k);
            if i2 >= 1 && i2 <= N_fast
                i0 = floor(i2); frac = i2-i0;
                if i0 < 1
                    v2s = 0;
                elseif i0+1> N_fast
                    v2s = s_tx_chirp(end);
                else
                    v2s = (1-frac)*s_tx_chirp(i0) + frac*s_tx_chirp(i0+1);
                end
                s_rx2(k) = alpha2 * v2s;
            end
        end
        s_rx = s_rx1 + s_rx2;
        % receive noise
        rx_power = mean(abs(s_rx).^2);
        if rx_power==0, rx_power=1e-12; end
        noise_rx = sqrt(rx_power / SNR_lin / 2) * (randn(size(s_rx)) + 1j*randn(size(s_rx)));
        s_rx_noisy = s_rx + noise_rx;
        % transmit noisy (optional small noise)
        tx_noise = sqrt(mean(abs(s_tx_chirp).^2)/SNR_lin/2) * (randn(size(s_tx_chirp)) + 1j*randn(size(s_tx_chirp)));
        s_tx_noisy = s_tx_chirp + tx_noise;
        % mix
        beat = s_tx_noisy .* conj(s_rx_noisy);
        BEAT_mat(:,m) = beat(:);
    end
    % ----------------------------
    % 2D FFT (Range-Doppler)：先 range，再 doppler
    % ----------------------------
    Nfft_range = 4096;
    Nfft_doppler = 256;
    win_range = hann(N_fast);
    win_doppler = hann(num_chirps);
    BEAT_win = (BEAT_mat .* repmat(win_range(:),1,num_chirps)) .* repmat(win_doppler(:).', N_fast, 1);
    R_fft = fft(BEAT_win, Nfft_range, 1);
    RD_map = fftshift(fft(R_fft, Nfft_doppler, 2), 2);
    RD_dB = 20*log10(abs(RD_map)+eps);
    % ----------------------------
    % Range axis 转换、截取前半（正频）
    % ----------------------------
    df_range = fs / Nfft_range;
    f_axis = (0:Nfft_range-1)*df_range;
    R_axis = (c * f_axis) / (2 * S);    % m
    half_rng = 1:floor(Nfft_range/2);
    R_use = R_axis(half_rng);
    RD_use = RD_dB(half_rng, :);
    % ----------------------------
    % 绘图
    % ----------------------------
    subplot(1, length(B_list), ib);
    imagesc((-Nfft_doppler/2:Nfft_doppler/2-1)*(1/(PRI*Nfft_doppler))*lambda/2, R_use, RD_use);
    axis xy;
    xlabel('Velocity (m/s)'); ylabel('Range (m)');
    title(sprintf('B = %.0f MHz', B/1e6));
    colormap jet; caxis(max(RD_use(:)) + [-40 0]);
    colorbar; hold on;
    % 标注真实目标位置
    plot([ -1 1 ], [R1 R1], 'w--','LineWidth',1);
    plot([ -1 1 ], [R2 R2], 'w--','LineWidth',1);
    hold off;
    % ----------------------------
    % 理论与测量的距离分辨率
    % ----------------------------
    theo_DR(ib) = c / (2 * B);   % 理论距离分辨率
    % 测量：在 range 方向找到两峰并估算其 3dB 宽度（以米为单位）
    % 简单方法：取目标 R 处的列（doppler 中心列），沿 range 做 1D 峰宽测量
    dop_col = round(Nfft_doppler/2); % 零多普勒列
    range_profile = RD_use(:, dop_col);
    % 找两个峰的索引（寻找两个局部最大值接近 R1,R2）
    [pks, locs] = findpeaks(range_profile, 'NPeaks', 4, 'SortStr','descend');
    if isempty(locs)
        meas_DR(ib) = NaN;
    else
        % 取最接近 R_target 的峰作为参考
        [~, idx_closest] = min(abs(R_use(locs) - R_target));
        loc_peak = locs(idx_closest);
        peak_val = range_profile(loc_peak);
        half_power = peak_val - 3;
        % find left/right indices where profile crosses half_power
        left_idx = find(range_profile(1:loc_peak) <= half_power, 1, 'last');
        if isempty(left_idx), left_idx = 1; end
        right_idx = find(range_profile(loc_peak:end) <= half_power, 1, 'first');
        if isempty(right_idx), right_idx = length(range_profile)-loc_peak+1; end
        right_idx = right_idx + loc_peak - 1;
        fwhm_bins = right_idx - left_idx;
        meas_DR(ib) = fwhm_bins * (R_use(2)-R_use(1)); % 转为米
    end
end

% ----------------------------
% 打印量化结果
% ----------------------------
fprintf('B (GHz)   Theoretical ΔR (m)    Measured FWHM (m)\n');
for ib = 1:length(B_list)
    fprintf('%.3f       %.6f             %.6f\n', B_list(ib)/1e9, theo_DR(ib), meas_DR(ib));
end
