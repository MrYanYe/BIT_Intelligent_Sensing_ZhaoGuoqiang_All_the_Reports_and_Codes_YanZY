%《调频连续波雷达原理、设计及应用》编程例子
% LFM信号匹配滤波方法
clc; clear; close all;
B=1.5e3;
fs=10*B;
kr=1e4;
T=B/kr;
t=-T/2:1/fs:T/2;
sig=exp(1j*pi*kr*t.^2);
plot(t/T, real(sig),'linewidth',1)
ylabel('Amplitude (V)')
xlabel('Normalized time');

sig_f=fftshift(fft(sig));
N=length(sig);
f=-fs/2:fs/N:fs/2-fs/N;
f=f/fs;
figure;plot(f,abs(sig_f),'linewidth',1)
ylabel('Amplitude')
xlabel('Normalized freuqncy');

len1=128;
figure;stft(sig,fs,'Window',kaiser(len1,5),'OverlapLength',len1/4,'FFTLength',len1);
h=colorbar;a = h.Position;title(h,'dB/Hz');
set(h,'Position',[a(1)+0.08 a(2)+0.05 0.01 0.6]);


sig_mf=conv(sig, conj(sig));
sig_mf=sig_mf/max(abs(sig_mf));
figure;
plot((abs(((sig_mf)))));


c0=3e8;
fc=10e9;
Rmax=1e8;
tdelay=2*Rmax/c0;
t1=0:1/fs:tdelay-1/fs;
echo_num=length(t1);
echo(1:echo_num)=10*(rand(1,echo_num)+1i*rand(1,echo_num));

target_range=Rmax*[0.2,0.39,0.5,0.6,0.68];
target_rcs=[2,3,5,3,2];
target_delay=2*target_range/c0;
target_delay_pos=round(target_delay*fs);

for tn=1:length(target_range)
    idx=target_delay_pos(tn):target_delay_pos(tn)+length(sig)-1;
    echo(idx) = echo(idx)+target_rcs(tn)*sig*exp(-1i*4*pi*fc*target_range(tn)/c0);%/ (target_range(tn)^4);
end
    

idR=1e-7*c0/2*(0:1/fs:(echo_num-1)/fs);
echo_mf=conv(echo,conj(sig),'same');
echo_mf=echo_mf/max(abs(echo_mf));

fontsz=12;
figure;plot(idR-(1e-7*T*c0/4),abs(echo_mf));
xlabel('相对位置');ylabel('相对幅度');
set(gcf,'color',[1 1 1]);
set(gca,'FontSize',fontsz)
set(findall(gcf,'type','text'),'FontSize',fontsz);

echo_fft=fft(echo,echo_num);
mh_fft=conj(fft(sig,echo_num));

match_out=ifft(echo_fft.*mh_fft);
match_out=match_out/max(abs(match_out));

figure;plot(idR,abs(match_out));
xlabel('相对位置');ylabel('相对幅度');
set(gcf,'color',[1 1 1]);
set(gca,'FontSize',fontsz)
set(findall(gcf,'type','text'),'FontSize',fontsz);