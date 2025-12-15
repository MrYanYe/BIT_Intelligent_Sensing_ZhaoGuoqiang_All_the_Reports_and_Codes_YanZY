%% FMCW_TwoTargets_Animated_Fixed2.m
close all; clear; clc;

% ---------- 参数 ----------
c   = 3e8; fc = 77e9; B = 2e9;
Tc  = 40e-6; fs = 4e9; S = B/Tc;
SNR_dB = 30;
dt = 1/fs;
t = 0:dt:Tc-dt;
N = length(t);

% 两目标
R1=10; v1=5; s1=0.8;
R2=15; v2=-3; s2=0.6;
tau1 = 2*R1/c; tau2 = 2*R2/c;
lambda = c/fc; fD1 = 2*v1/lambda; fD2 = 2*v2/lambda;
alpha1 = s1/(R1^2); alpha2 = s2/(R2^2);

% 发射 chirp
phi = 2*pi*(0.*t + 0.5*S.*t.^2);
s_tx = exp(1j*phi);

% 延时插值生成回波
delay1 = tau1*fs; delay2 = tau2*fs;
s_rx1 = zeros(1,N); s_rx2 = zeros(1,N);
n = 0:N-1;
t_idx1 = n - delay1; t_idx2 = n - delay2;
for k=1:N
    idx = t_idx1(k);
    if idx>=1 && idx<=N
        i0 = floor(idx); frac = idx - i0;
        if i0<1, val=0;
        elseif i0+1> N, val = s_tx(end);
        else val = (1-frac)*s_tx(i0) + frac*s_tx(i0+1); end
        s_rx1(k) = alpha1 * val * exp(1j*2*pi*fD1*((k-1)/fs - tau1));
    end
    idx = t_idx2(k);
    if idx>=1 && idx<=N
        i0 = floor(idx); frac = idx - i0;
        if i0<1, val=0;
        elseif i0+1> N, val = s_tx(end);
        else val = (1-frac)*s_tx(i0) + frac*s_tx(i0+1); end
        s_rx2(k) = alpha2 * val * exp(1j*2*pi*fD2*((k-1)/fs - tau2));
    end
end
s_rx = s_rx1 + s_rx2;
% 噪声
rx_power = mean(abs(s_rx).^2);
SNR_lin = 10^(SNR_dB/10);
noise_p = max(rx_power/SNR_lin,1e-18);
s_rx_noisy = s_rx + sqrt(noise_p/2)*(randn(size(s_rx))+1j*randn(size(s_rx)));
s_tx_noisy = s_tx + sqrt(mean(abs(s_tx).^2)/SNR_lin/2)*(randn(size(s_tx))+1j*randn(size(s_tx)));

% beat 全段
beat = s_tx_noisy .* conj(s_rx_noisy);

% 瞬时频率
phi_un = unwrap(angle(s_tx)); f_inst = [0,diff(phi_un)/(2*pi*dt)]; f_inst_MHz = f_inst/1e6;

% 瀑布与底图参数
waterfall_Nfft = 2^10; win_len = max(64,round(0.05*N)); hop = max(1,round(win_len/8));
waterfall_maxCols = 300;
Nfft_full = 2^14;
freq_axis_full = (-Nfft_full/2 : Nfft_full/2-1) * (fs / Nfft_full);

% 画布
figure('Name','FMCW Two-target test animation (fixed2)','NumberTitle','off','Position',[100 100 1200 800]);
ax1 = subplot(3,3,[1 2]); h_tx=plot(ax1,nan,nan,'b'); hold(ax1,'on'); h_r1=plot(ax1,nan,nan,'r'); h_r2=plot(ax1,nan,nan,'m');
h_sum=plot(ax1,nan,nan,'k'); h_beat=plot(ax1,nan,nan,'g'); legend(ax1,'tx','rx1','rx2','sum','beat'); xlabel(ax1,'Time (\mus)'); ylabel(ax1,'Amp'); xlim(ax1,[0 Tc*1e6]); grid(ax1,'on');
ax2 = subplot(3,3,3); h_inst=plot(ax2,nan,nan,'r'); xlabel(ax2,'Time (\mus)'); ylabel(ax2,'Freq (MHz)'); title(ax2,'Inst freq'); xlim(ax2,[0 Tc*1e6]); ylim(ax2,[min(f_inst_MHz) max(f_inst_MHz)]);
ax3 = subplot(3,3,[4 6]); fvec = (-waterfall_Nfft/2:waterfall_Nfft/2-1)*(fs/waterfall_Nfft); hImg = imagesc(ax3,[1 1],fvec/1e3,zeros(length(fvec),1)); axis(ax3,'xy'); colormap(ax3,'jet'); colorbar(ax3); xlabel(ax3,'Frame'); ylabel(ax3,'Freq (kHz)'); caxis(ax3,[-120 -30]);
ax4 = subplot(3,3,7:9); h_spec=plot(ax4,nan,nan,'k'); hold(ax4,'on'); h_peak=plot(ax4,nan,nan,'ro'); xlabel(ax4,'Freq (kHz)'); ylabel(ax4,'Mag (dB)'); title(ax4,'Full-chirp FFT'); grid(ax4,'on'); xlim(ax4,[0 fs/2/1e3]);

% 瀑布数据容器
imgData = []; frameIdx=0;
ds = 10; % 显示下采样

% 主循环（逐样点展示）
for k=1:N
    t_now = t(1:k); tx_now = s_tx_noisy(1:k); r1_now = s_rx1(1:k); r2_now = s_rx2(1:k); sum_now = s_rx_noisy(1:k); beat_now = beat(1:k);
    idx_plot = 1:ds:k;
    set(h_tx,'XData',t_now(idx_plot)*1e6,'YData',real(tx_now(idx_plot)));
    set(h_r1,'XData',t_now(idx_plot)*1e6,'YData',real(r1_now(idx_plot)));
    set(h_r2,'XData',t_now(idx_plot)*1e6,'YData',real(r2_now(idx_plot)));
    set(h_sum,'XData',t_now(idx_plot)*1e6,'YData',real(sum_now(idx_plot)));
    set(h_beat,'XData',t_now(idx_plot)*1e6,'YData',real(beat_now(idx_plot)));
    set(h_inst,'XData',t_now*1e6,'YData',f_inst_MHz(1:k));
    drawnow limitrate;

    % 更新瀑布（按 hop）
    if k>=win_len && mod(k,hop)==0
        frameIdx = frameIdx +1;
        xwin = beat(k-win_len+1:k) .* hann(win_len).';
        X = fftshift(fft(xwin,waterfall_Nfft));
        PSDst = 20*log10(abs(X)+eps); PSDst = PSDst(:);
        if frameIdx==1, imgData = PSDst; else imgData = [imgData, PSDst]; end
        if size(imgData,2) > waterfall_maxCols, imgData = imgData(:,end-waterfall_maxCols+1:end); end
        set(hImg,'CData',imgData); set(hImg,'XData',[1 size(imgData,2)]); set(hImg,'YData',fvec/1e3);
        % 动态 caxis
        pmax = max(imgData(:)); caxis(ax3,[pmax-50 pmax]);
    end

    % 周期性全谱更新（较慢）
    if mod(k,round(0.02*N))==0 || k==N
        BEAT_FFT_now = fftshift(fft(beat .* hann(length(beat)).', Nfft_full));
        PSD_full = 20*log10(abs(BEAT_FFT_now)+eps);
        % 调试检查：若 PSD_full 包含非有限值则打印并跳过更新
        if any(isnan(PSD_full)) || any(isinf(PSD_full))
            warning('PSD_full contains NaN/Inf at k=%d — skip update', k);
        else
            half_idx = Nfft_full/2+1 : Nfft_full;
            freq_pos = freq_axis_full(half_idx)/1e3;
            PSD_pos = PSD_full(half_idx);
            % 检查长度一致性
            if numel(freq_pos)==numel(PSD_pos)
                set(h_spec,'XData',freq_pos,'YData',PSD_pos);
                % 峰值标注
                [~,mx] = max(PSD_pos);
                f_peak = freq_pos(mx)*1e3; % Hz
                set(h_peak,'XData',freq_pos(mx),'YData',PSD_pos(mx));
                % 计算并打印距离估计（演示）
                R_est = c * f_peak / (2*S);
                % 在底图注释（移除旧 Tag）
                old = findobj(ax4,'Tag','estText'); delete(old);
                text(ax4,freq_pos(mx),PSD_pos(mx),sprintf(' R=%.2f m',R_est),'Color','r','Tag','estText','VerticalAlignment','bottom','HorizontalAlignment','left');
            else
                warning('freq_pos and PSD_pos length mismatch');
            end
        end
    end
end

% 最后打印测试信息
disp('Done. If bottom plot is still empty, run the debug checks described earlier.');
