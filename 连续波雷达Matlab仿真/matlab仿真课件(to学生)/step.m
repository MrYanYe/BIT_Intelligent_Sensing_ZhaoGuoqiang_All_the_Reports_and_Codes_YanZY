clear all
close all

% ===== 波形参数
c = 3e8; % 光速
tau = 0.16e-6;
ts = tau / 4;
fs = 1 / ts;
delt_f = 8e6; % 步进频率
f0 = 35e9; % 载频
N = 64; % 步进脉冲数目
PRT = 25e-6;
% ===== 目标参数
R_tgt = [5 15 3];
RCS = [1, 3.4, 2];
V_tgt = [0, 0, 0];
N_tgt = length(R_tgt);
N_tgt = 3;

% ===== 仿真参数
t_rec = 0:ts:PRT; % 接受窗长度为一个脉冲重复周期
len_rec = length(t_rec);
t_tau = (1:round(tau*fs))/fs;


% ========== 数字仿真生成回波后进行混频
sig_updn = zeros(2 * N, len_rec);

% ===== 按照去载频后的信号进行仿真（直接生成混频后的回波信号
% === 上变频信号
sig_up_mix = zeros(N, len_rec);
for i = 1:N
    % 目标循环
    tar_temp = zeros(1, len_rec);
    for j = 1:N_tgt
        R = R_tgt(j);
        v = V_tgt(j);
        taur = 2 * R / c - 2 * v * (i - 1) * PRT / c; % 前面的项为目标固定时延，后面为速度造成的时延
        tp = c * tau / (c - v); % 根据目标速度修正脉宽值
        temp = RCS(j) * rectpuls(t_rec - tp / 2 - taur, tp / 2).* ...
                    exp(-2j * pi * (f0 + (i - 1) * delt_f) * taur);
        tar_temp = tar_temp + temp; % 更新当前帧目标的回波信号
    end
    sig_up_mix(i, :) = sig_up_mix(i, :) + tar_temp;
end
% plot(real(sig_up_mix(4, :)));
% mesh(abs(sig_up_mix));
% === 下变频信号
% sig_dn_mix = zeros(N, len_rec);
% for i = 1:N
%     % 目标循环
%     tar_temp = zeros(1, len_rec);
%     for j = 1:N_tgt
%         R = R_tgt(j);
%         v = V_tgt(j);
%         taur = 2 * R / c - 2 * v * (i) * PRT / c; % 前面的项为目标固定时延，后面为速度造成的时延
%         tp = c * tau / (c - v); % 根据目标速度修正脉宽值
%         temp = RCS(j) * rectpuls(t_rec - tp / 2 - taur, tp / 2).* ...
%                     exp(-2j * pi * (f0 + (N - i) * delt_f) * taur);
%         tar_temp = tar_temp + temp;
%     end
%     sig_dn_mix(i, :) = sig_dn_mix(i, :) + tar_temp;
% end
% % === 生成总回波信号
% sig_ud_mix = [sig_up_mix; sig_dn_mix];


% ===== 试着先做一次IFFT 检验论文中的IFFT是将上下分开进行的还是一起进行的
% sig_ud_ifft = zeros(2 * N, len_rec);
% for i = 1:len_rec
%     sig_ud_ifft(:, i) = ifft(sig_ud_mix(:, i));
% end
%%============================= 注意！下变频信号要翻转读取！！！！=====================
sig_up_ifft = ifft(sig_up_mix, [], 1);
% sig_dn_ifft = ifft(flipud(sig_dn_mix), [], 1); % ！！！要对下变频进行逆序翻转
% sig_ud_ifft_sp = [sig_up_ifft; sig_dn_ifft]; % 分开的ifft
delta_r = c/(2*N * delt_f);
ri = c/(2 * delt_f);
figure
plot(0:delta_r:ri - delta_r, abs(sig_up_ifft(:, 6)));
% hold on
% plot(0:delta_r:ri - delta_r, abs(sig_dn_ifft(:, 20)));


figure
mesh( 1:626, 0:delta_r:ri - delta_r, abs(sig_up_ifft));


