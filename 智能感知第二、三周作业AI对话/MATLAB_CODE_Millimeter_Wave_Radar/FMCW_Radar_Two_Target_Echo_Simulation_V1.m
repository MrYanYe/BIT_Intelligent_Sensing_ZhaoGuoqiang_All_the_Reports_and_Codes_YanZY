%% 两目标仿真 - 在你原脚本基础上插入或替换以下代码块
close all;
clc; clear;

% 基本参数（与原脚本一致）
c = 3e8;
fc = 77e9;
B  = 2e9;
Tc = 40e-6;
fs = 4e9;
S  = B / Tc;
f0 = 0;
SNR_dB = 30;

dt = 1/fs;
t = 0:dt:Tc-dt;
phi = 2*pi*( f0.*t + 0.5 * S .* t.^2 );
s_bb = exp(1j*phi);

% 目标参数：两个目标
R1 = 10; v1 = 5;    sigma1 = 0.8;
R2 = 15; v2 = -3;   sigma2 = 0.6;

% 计算时延与多普勒
tau1 = 2*R1 / c;
tau2 = 2*R2 / c;
lambda = c / fc;
fD1 = 2*v1 / lambda;
fD2 = 2*v2 / lambda;

% 衰减系数（工程近似）
alpha1 = sigma1 * (1 / (R1^2));
alpha2 = sigma2 * (1 / (R2^2));

% 将发射信号延时（亚样点）并加上多普勒调制
N = length(t);
delay_samples1 = tau1 * fs;
delay_samples2 = tau2 * fs;
s_rx1 = zeros(size(s_bb));
s_rx2 = zeros(size(s_bb));

% 线性插值实现亚样点延迟并叠加多普勒相位
n = 0:N-1;
t_idx_shift1 = n - delay_samples1;
t_idx_shift2 = n - delay_samples2;
for k = 1:N
    idx1 = t_idx_shift1(k);
    if idx1 >= 1 && idx1 <= N
        i0 = floor(idx1);
        frac = idx1 - i0;
        if i0 < 1
            val1 = 0;
        elseif i0+1 > N
            val1 = s_bb(end);
        else
            val1 = (1-frac)*s_bb(i0) + frac*s_bb(i0+1);
        end
        % 多普勒项以 t - tau 相位乘上
        s_rx1(k) = alpha1 * val1 * exp(1j*2*pi*fD1*( (k-1)/fs - tau1 ));
    else
        s_rx1(k) = 0;
    end

    idx2 = t_idx_shift2(k);
    if idx2 >= 1 && idx2 <= N
        i0 = floor(idx2);
        frac = idx2 - i0;
        if i0 < 1
            val2 = 0;
        elseif i0+1 > N
            val2 = s_bb(end);
        else
            val2 = (1-frac)*s_bb(i0) + frac*s_bb(i0+1);
        end
        s_rx2(k) = alpha2 * val2 * exp(1j*2*pi*fD2*( (k-1)/fs - tau2 ));
    else
        s_rx2(k) = 0;
    end
end

% 合成总回波并加噪声
s_rx_total = s_rx1 + s_rx2;
% 接收噪声按接收功率与所需 SNR 比例
rx_power = mean(abs(s_rx_total).^2);
SNR_lin = 10^(SNR_dB/10);
if rx_power == 0; rx_power = 1e-12; end
noise_power_rx = rx_power / SNR_lin;
noise_rx = sqrt(noise_power_rx/2) * (randn(size(s_rx_total)) + 1j*randn(size(s_rx_total)));
s_rx_noisy = s_rx_total + noise_rx;

% 发射信号也可以加入噪声以保持一致显示（可选）
tx_noise = sqrt((mean(abs(s_bb).^2))/SNR_lin/2) * (randn(size(s_bb)) + 1j*randn(size(s_bb)));
s_bb_noisy = s_bb + tx_noise;

% 混频得到 beat 信号
beat = s_bb_noisy .* conj(s_rx_noisy);

% FFT 参数与谱计算
Nfft = 2^16;
w = hann(length(beat)).';
beat_w = beat .* w;
BEAT_FFT = fftshift(fft(beat_w, Nfft));
freq_axis = (-Nfft/2 : Nfft/2-1) * (fs / Nfft);
PSD = 20*log10(abs(BEAT_FFT) + eps);

% 寻峰：可能有两个显著峰
% 我们只看正频率部分以对应正的 beat（取绝对值处理后找局部峰）
half = Nfft/2+1:Nfft;
[~, pk_idx_sorted] = sort(abs(BEAT_FFT(half)),'descend');
topk = 4;
peaks = half(pk_idx_sorted(1:topk));
% 去重与筛选合理峰（这里挑两最大峰）
peak1 = peaks(1);
peak2 = peaks(2);
f_est1 = abs(freq_axis(peak1));
f_est2 = abs(freq_axis(peak2));

% 理论 beat 频率（包含多普勒）
fB1_theory = S * tau1 + fD1;
fB2_theory = S * tau2 + fD2;

% 距离估计去除多普勒影响：若已知速度，可用 fB - fD 恢复 S·tau，再算R
R1_est = c*(f_est1 - fD1) / (2*S);
R2_est = c*(f_est2 - fD2) / (2*S);

% 输出
fprintf('theory fB1 = %.3f Hz, theory fB2 = %.3f Hz\n', fB1_theory, fB2_theory);
fprintf('estimated peaks f1 = %.3f Hz, f2 = %.3f Hz\n', f_est1, f_est2);
fprintf('after removing fD: R1_est = %.6f m, R2_est = %.6f m\n', R1_est, R2_est);

% 绘图展示（保留或单独查看）
figure('Name','Two-target simulation','NumberTitle','off','Position',[200 200 1000 800]);

subplot(4,1,1);
plot(t*1e6, real(s_rx_noisy));
xlabel('Time (μs)'); ylabel('Re\{s_{rx}\}');
title(sprintf('Received composite echo (real). tau1=%.3fus tau2=%.3fus', tau1*1e6, tau2*1e6));
grid on;

subplot(4,1,2);
plot(t*1e6, real(beat));
xlabel('Time (μs)'); ylabel('Re\{beat\}');
title('Mixed signal (time domain) - composite');
grid on;

subplot(4,1,3);
% zoom to near arrival of earlier target
start_idx = max(1, round(delay_samples1) - 50);
end_idx = min(N, start_idx + 500);
plot(t(start_idx:end_idx)*1e6, real(beat(start_idx:end_idx)));
xlabel('Time (μs)'); ylabel('Re\{beat\}');
title('Beat signal zoom (around first echo arrival)');
grid on;

subplot(4,1,4);
plot(freq_axis/1e3, PSD, 'k'); hold on;
% 标注两个峰
plot(freq_axis(peak1)/1e3, PSD(peak1), 'ro', 'MarkerFaceColor','r');
text(freq_axis(peak1)/1e3, PSD(peak1), sprintf(' (%.3f kHz, %.2f dB)', freq_axis(peak1)/1e3, PSD(peak1)), ...
    'VerticalAlignment','bottom','HorizontalAlignment','left','Color','r');
plot(freq_axis(peak2)/1e3, PSD(peak2), 'mo', 'MarkerFaceColor','m');
text(freq_axis(peak2)/1e3, PSD(peak2), sprintf(' (%.3f kHz, %.2f dB)', freq_axis(peak2)/1e3, PSD(peak2)), ...
    'VerticalAlignment','bottom','HorizontalAlignment','left','Color','m');
xlabel('Frequency (kHz)'); ylabel('Magnitude (dB)');
title('FFT of beat (composite) - two target peaks annotated');
xlim([0 fs/2/1e3]);
grid on;
hold off;
