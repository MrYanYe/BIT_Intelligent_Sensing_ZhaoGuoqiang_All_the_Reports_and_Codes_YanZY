%% FMCW_TwoTargets_Animated.m
% Self-contained FMCW two-target animation:
% - show transmit chirp generation
% - show two delayed Doppler-shifted echoes arriving and overlapping
% - show real-time beat signal, short-time FFT waterfall
% - periodically run a simple frequency-peak based range estimate and annotate

close all; clear; clc;

%% ---------- 参数配置 ----------
c   = 3e8;
fc  = 77e9;
B   = 2e9;
Tc  = 40e-6;        % 单个 chirp 时长，可改为 100e-6 观察差别
fs  = 4e9;          % 采样率（调试时可下调，例如 1e8，加速显示）
S   = B / Tc;
f0  = 0;
SNR_dB = 30;

% 目标设置（可修改）
R1 = 10;   v1 = 5;    sigma1 = 0.8;
R2 = 15;   v2 = -3;   sigma2 = 0.6;

% 用于动画控制
display_downsample = 20;   % 绘图时沿时间轴的下采样，否则点太多（整数）
waterfall_Nfft = 2^11;     % 短时 FFT 长度（实时展示用）
waterfall_maxCols = 300;   % 瀑布图最大列数
iter_update_interval = round(0.01 * fs); % 每多少采样点做一次频域估计显示（例如占 chirp 的 1%）

%% ---------- 信号与回波生成（离线准备） ----------
dt = 1/fs;
t = 0:dt:Tc-dt;                % 单 chirp 时间轴
N = length(t);

% 发射基带 chirp（相位）
phi = 2*pi*( f0.*t + 0.5 * S .* t.^2 );
s_tx = exp(1j*phi);

% 目标时延与多普勒
tau1 = 2*R1 / c;
tau2 = 2*R2 / c;
lambda = c / fc;
fD1 = 2*v1 / lambda;
fD2 = 2*v2 / lambda;

% 衰减
alpha1 = sigma1 * (1 / (R1^2));
alpha2 = sigma2 * (1 / (R2^2));

% 计算延时的样点（亚样点）
delay_samps1 = tau1 * fs;
delay_samps2 = tau2 * fs;

% 预分配接收向量（纯实时间序列，含两目标回波）
s_rx_total = zeros(1, N);
s_rx1 = zeros(1,N); s_rx2 = zeros(1,N);

% 用线性插值实现亚样点延时并乘多普勒相位（时域再生）
n = 0:N-1;
t_idx_shift1 = n - delay_samps1;
t_idx_shift2 = n - delay_samps2;
for k = 1:N
    idx1 = t_idx_shift1(k);
    if idx1 >= 1 && idx1 <= N
        i0 = floor(idx1);
        frac = idx1 - i0;
        if i0 < 1
            val1 = 0;
        elseif i0+1 > N
            val1 = s_tx(end);
        else
            val1 = (1-frac)*s_tx(i0) + frac*s_tx(i0+1);
        end
        s_rx1(k) = alpha1 * val1 * exp(1j*2*pi*fD1*( (k-1)/fs - tau1 ));
    end

    idx2 = t_idx_shift2(k);
    if idx2 >= 1 && idx2 <= N
        i0 = floor(idx2);
        frac = idx2 - i0;
        if i0 < 1
            val2 = 0;
        elseif i0+1 > N
            val2 = s_tx(end);
        else
            val2 = (1-frac)*s_tx(i0) + frac*s_tx(i0+1);
        end
        s_rx2(k) = alpha2 * val2 * exp(1j*2*pi*fD2*( (k-1)/fs - tau2 ));
    end
end
s_rx_total_clean = s_rx1 + s_rx2;

% 添加接收噪声
rx_power = mean(abs(s_rx_total_clean).^2);
SNR_lin = 10^(SNR_dB/10);
noise_p = max(rx_power / SNR_lin, 1e-18);
noise_rx = sqrt(noise_p/2) * (randn(size(s_rx_total_clean)) + 1j*randn(size(s_rx_total_clean)));
s_rx_total = s_rx_total_clean + noise_rx;

% 发射加噪（用于混频）
tx_noise = sqrt(mean(abs(s_tx).^2)/SNR_lin/2) * (randn(size(s_tx)) + 1j*randn(size(s_tx)));
s_tx_noisy = s_tx + tx_noise;

% 混频得到 beat（整段）
beat_full = s_tx_noisy .* conj(s_rx_total);

%% ---------- 画布布局（多子图，实时更新） ----------
fig = figure('Name','FMCW Two-target live animation','NumberTitle','off','Position',[50 50 1400 800]);

ax1 = subplot(3,3,[1 2]); % 发射与接收时域（顶部宽图）
h_tx = plot(ax1, nan, nan, 'b', 'DisplayName','tx (real)'); hold(ax1,'on');
h_rx1 = plot(ax1, nan, nan, 'r', 'DisplayName','rx target1 (real)');
h_rx2 = plot(ax1, nan, nan, 'm', 'DisplayName','rx target2 (real)');
h_rxSum = plot(ax1, nan, nan, 'k', 'DisplayName','rx sum (real)');
h_beat = plot(ax1, nan, nan, 'g', 'DisplayName','beat (real)');
legend(ax1,'show'); xlabel(ax1,'Time (\mus)'); ylabel(ax1,'Amplitude');
title(ax1,'Tx, Rx echoes and beat (time domain)'); grid(ax1,'on');

ax2 = subplot(3,3,3); % 瞬时频率
h_inst = plot(ax2, nan, nan, 'r'); grid(ax2,'on');
xlabel(ax2,'Time (\mus)'); ylabel(ax2,'Inst freq (MHz)'); title(ax2,'Instantaneous frequency of TX');

ax3 = subplot(3,3,[4 6]); % 短时 FFT 瀑布
hImg = imagesc(ax3, 1, linspace(-fs/2, fs/2, waterfall_Nfft)/1e3, zeros(waterfall_Nfft,1));
axis(ax3,'xy'); colormap(ax3,'jet'); colorbar(ax3);
xlabel(ax3,'Frame index'); ylabel(ax3,'Freq (kHz)');
title(ax3,'Short-time FFT waterfall');

ax4 = subplot(3,3,7:9); % 频谱与估计
h_spec = plot(ax4, nan, nan, 'k'); hold(ax4,'on');
h_peak = plot(ax4, nan, nan, 'ro','MarkerFaceColor','r');
xlabel(ax4,'Freq (kHz)'); ylabel(ax4,'Mag (dB)');
title(ax4,'Full-chirp FFT (periodic update)'); grid(ax4,'on');

% 预计算瞬时频率用于显示
phi_un = unwrap(angle(s_tx));
f_inst = [0, diff(phi_un)/(2*pi*dt)];
f_inst_MHz = f_inst / 1e6;

% waterfall fft axis
fvec = (-waterfall_Nfft/2 : waterfall_Nfft/2-1) * (fs / waterfall_Nfft);

% 历史数据容器
imgData = [];
frameIdx = 0;

% Nfft 用于周期性全谱估计显示
Nfft_full = 2^14;
freq_axis_full = (-Nfft_full/2 : Nfft_full/2-1) * (fs / Nfft_full);

%% ---------- 动画主循环：逐采样推进，实时绘图 ----------
% 为减少绘图负担，实际绘制使用 downsample 显示点
ds = max(1, display_downsample);

% 迭代估计记录
estRanges = []; estFreqs = [];

for k = 1:N
    % 当前时间窗口（从 1 到 k）
    t_now = t(1:k);
    tx_now = s_tx_noisy(1:k);
    rx1_now = s_rx1(1:k);
    rx2_now = s_rx2(1:k);
    rxsum_now = s_rx_total(1:k);
    beat_now = beat_full(1:k);

    % 显示：时域（下采样）
    idx_plot = 1:ds:k;
    set(h_tx, 'XData', t_now(idx_plot)*1e6, 'YData', real(tx_now(idx_plot)));
    set(h_rx1, 'XData', t_now(idx_plot)*1e6, 'YData', real(rx1_now(idx_plot)));
    set(h_rx2, 'XData', t_now(idx_plot)*1e6, 'YData', real(rx2_now(idx_plot)));
    set(h_rxSum, 'XData', t_now(idx_plot)*1e6, 'YData', real(rxsum_now(idx_plot)));
    set(h_beat, 'XData', t_now(idx_plot)*1e6, 'YData', real(beat_now(idx_plot)));

    % 瞬时频率随时间显示（仅显示到当前时刻）
    set(h_inst, 'XData', t_now*1e6, 'YData', f_inst_MHz(1:k));
    xlim(ax2, [0 Tc*1e6]);

    % 滑动短时 FFT：当当前点数足够形成一个窗时更新一列（hop = win_len/8）
    win_len = max(64, round(0.05 * N));  % 固定窗长度的近似（可调）
    hop = max(1, round(win_len/8));
    if k >= win_len && mod(k, hop) == 0
        frameIdx = frameIdx + 1;
        xwin = beat_full(k-win_len+1:k) .* hann(win_len).';
        X = fftshift(fft(xwin, waterfall_Nfft));
        PSDst = 20*log10(abs(X)+eps);
        if frameIdx == 1
            imgData = PSDst;
        else
            imgData = [imgData, PSDst]; %#ok<AGROW>
            if size(imgData,2) > waterfall_maxCols
                imgData = imgData(:, end-waterfall_maxCols+1:end);
            end
        end
        set(hImg, 'CData', imgData);
        set(hImg, 'XData', [1 size(imgData,2)]);
        set(hImg, 'YData', fvec/1e3);
    end

    % 每隔若干采样做一次全谱频域估计并标注峰值（更慢的更新频率）
    if mod(k, iter_update_interval) == 0 || k==N
        BEAT_FFT_now = fftshift(fft(beat_full .* hann(length(beat_full)).', Nfft_full));
        PSD_full = 20*log10(abs(BEAT_FFT_now)+eps);
        set(h_spec, 'XData', freq_axis_full/1e3, 'YData', PSD_full);
        % 取正频段最大峰并标注
        half_idx = Nfft_full/2+1 : Nfft_full;
        [~, idx_sorted] = sort(abs(BEAT_FFT_now(half_idx)), 'descend');
        peak_idx_global = half_idx(idx_sorted(1));
        f_peak = abs(freq_axis_full(peak_idx_global));
        set(h_peak, 'XData', f_peak/1e3, 'YData', PSD_full(peak_idx_global));
        % 映射为距离（此处未去多普勒，仅演示）
        R_est = c * f_peak / (2*S);
        estRanges = [estRanges; R_est]; %#ok<AGROW>
        estFreqs = [estFreqs; f_peak]; %#ok<AGROW>
        % 在 ax4 显示估计文本
        text(ax4, f_peak/1e3, PSD_full(peak_idx_global), sprintf(' R=%.1f m', R_est), ...
            'VerticalAlignment','bottom','HorizontalAlignment','left','Color','r','FontSize',9);
    end

    drawnow limitrate;
    % 小 pause 带来更平滑的动画（可调整或移除以加速）
    % pause(0.0001);
end

%% ---------- 结尾显示最终估计 ----------
figure('Name','Final estimates','NumberTitle','off','Position',[200 200 600 300]);
plot(estRanges, 'o-'); grid on;
xlabel('Update index'); ylabel('Estimated Range (m)');
title('Iterative periodical range estimates');

fprintf('Final estimated ranges (m):\n');
disp(estRanges);
