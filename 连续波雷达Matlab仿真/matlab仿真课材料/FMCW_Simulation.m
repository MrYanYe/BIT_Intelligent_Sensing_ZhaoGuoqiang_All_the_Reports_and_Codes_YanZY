% Automotive Medium Range Radar (MRR)
% 对小汽车 RCS =10 sqm 的作用距离 160 m
% 分辨率 0.3 m
% 角度分辨率 8 度


clc;clear;close all;

c0   = 3e8; %Light Speed
% MRR Performance Requirements
delta_R=0.3;% m
delta_V= 1; % m/s
delta_angle=8;% degrees
SNR_min=10; % dB
R_max=160;% m
Vel_max =200;% km/h
RCS_car=10*log10(10);% dBsqm

% Step 1
Fc=78e9;
lambda=c0/Fc;

% Step 2
Na=round(360/pi/delta_angle);%天线阵元数量
delta_dis= lambda/2;  %天线阵元间距

Gt= 15; % dBi  Transmitted Anttena Gain
Gr=15;% dBi Recieved Anttena Gain

% Step 3
F= 15; % Noise figure in dB
Pt= 12; % Transmiited power dBm
L =2 ; % System loss dB

% Step 4
fd_max=2*Vel_max*1e3/3600/lambda;
T=1/(2*fd_max);% sweep time

PRF=1/T;
CPI=round(PRF/(2*delta_V/lambda));
CPI    = 2^(round(log2(CPI))); 

% Step 5
R_max_new=c0*T/2;

if R_max_new< R_max
    quit;
end

te = 290.0; % effective noise temperature in Kelvins
k_constant=1.38e-23;

Rmax_dB= Pt-30+Gt +Gr + 20*log10(lambda)+RCS_car+10*log10(CPI)+10*log10(T)...
 -30*log10(4*pi)-SNR_min-10*log10(k_constant*te)-F-L;%考虑了相关积累贡献

Rmax_radar_eq=(10^(Rmax_dB/10))^0.25;

if R_max_new<Rmax_radar_eq   %检验雷达最大作用距离是否在不模糊距离之内
    quit;
end

if  (Rmax_radar_eq-R_max)<0 %检验雷达最大作用距离是否达到设计要求
      quit;
end

% Step 6
B=c0/(2*delta_R);  %信号带宽
Kr=B/T;                %调频率

fs= 4*Kr*R_max/c0+1/T; % 基带信号采样频率
RF_fs= B*3;


t  = (0 : 1/RF_fs : T-1/RF_fs);
TX_RF   = exp(1i*pi*Kr*t.^2).*exp(1i*2*pi*Fc*t);
TX_Ref  = conj(TX_RF);

fs=RF_fs/round(RF_fs/fs);

N_Fast  = round(T*fs);  

dletaR=fs/N_Fast*3e8/2/Kr;
idR=dletaR*[0:N_Fast/2-1];
idV=lambda/2*(-PRF/2:PRF/CPI:PRF/2-PRF/CPI);
idt = 1e6*(0 : 1/fs : T);

%Lowpass Design
Fpass = 0.9*fs/2;        % Passband Frequency
Fstop =1.1*Fpass;        % Stopband Frequency
Dpass = 0.0057501127785;  % Passband Ripple
Dstop = 0.0001;          % Stopband Attenuation
dens  = 20;              % Density Factor

% Calculate the order from the parameters using firpmord
[N, Fo, Ao, W] = firpmord([Fpass, Fstop]/(fs/2), [1 0], [Dpass, Dstop]);

% Calculate the filter coefficients using the firpmord function.
b  = firpm(N, Fo, Ao, W, {dens});
Hd = dfilt.dffir(b);
fvtool(Hd,'Fs',fs);


target_rcs=[8,10,15];
target_range=[36, 50, 60];
target_velocity=[5, -10 ,15];
target_theta=[-35, 0, 60];

noise_amp=k_constant*te*10^(F/10)*B;
noise_amp=sqrt(noise_amp);%convert power gain to voltage gain

 gplot=0;  %用于画发射接收信号图的标识
 
 %https://e2e.ti.com/support/sensors-group/sensors/f/sensors-forum/...
 %719803/awr1642-confirm-power-in-tx-and-gain-in-rx
 LNA_ADC_Gain= 48; % dB   the LNA to ADC gain
 
 % Range Doppler Raw Data Simulation
 Rawdata(CPI,N_Fast)=0;
for k=1:CPI
    
    echo=TX_RF*0;

    for tn=1:length(target_range)
        range=target_range(tn)+target_velocity(tn)*T*(k-1);
        echo= echo+ target_rcs(tn)*exp(1i*pi*Kr*(t-2*range/c0).^2).*exp(1i*2*pi*Fc*(t-2*range/c0)) / (range^4);
    end
    
    Gain    = Pt-30+Gt +Gr + 20*log10(lambda)+LNA_ADC_Gain-30*log10(4*pi)-L;
    Gain    =sqrt(10^(Gain/10)); %convert power gain to voltage gain

    echo   = echo * Gain;

    echo=echo+ noise_amp.*(randn(1,length(echo))+1i*randn(1,length(echo))); 



    Mixer_Output = conj(echo.*TX_Ref); %混频


    Mixer_Output = decimate(Mixer_Output, RF_fs/fs);
    RX_Base  = filter(Hd, Mixer_Output); 

    Rawdata(k,:)=RX_Base;

    if(gplot)
        STFT_WindonwLen = 256;
        figure(1)
        set(gcf, 'Position', [0 0 1000 800])
        subplot(211)
        % Plot the STFT of the received signal before the mixer
        spectrogram(echo,STFT_WindonwLen,round(STFT_WindonwLen*0.8),STFT_WindonwLen, RF_fs, 'centered','yaxis');
        %ylim([-1 2])

        STFT_WindonwLen = 128;
        subplot(212)
        % Plot the STFT of the received signal after the mixer
        spectrogram(Mixer_Output,STFT_WindonwLen,round(STFT_WindonwLen*0.8),STFT_WindonwLen, fs, 'centered','yaxis');
        %ylim([-0.1 0.1])
        
        figure(2)
        subplot(211);
        plot(idt,1e3*real(RX_Base),'b','linewidth',1);        % Plot the received signal in the time domain
        %ylim([-5 5]);
        xlim([0 T]*1e6);
        xlabel('Time (\mus)');
        ylabel('Amplitude(mV)');
    
        subplot(212);
        plot(idt,1e3*abs(RX_Base),'b','linewidth',1);        % Plot the received signal in the time domain
        %ylim([-5 5]);
      
        legend('Magnitude','Envelope');
        xlim([0 T]*1e6);
        xlabel('Time (\mus)');
        ylabel('Amplitude(mV)');
    end
    
end


x=Rawdata(1,:);
N_sig = length(x);   %  length of signal

fontsz=16;

raw_data=Rawdata.';
raw_data=fftshift(fft(raw_data,[],1),1); % Range FFT
raw_data=fftshift(fft(raw_data,[],2),2); % FFT to Range Doppler Domain
raw_data=raw_data(1+N_sig/2:N_sig,:);

figure;
surf(idV,idR,20*log10(abs(raw_data)+eps)+30);
xlabel(['Velocity (','m/s)']);
ylabel('Range (m)');
zlabel('Intensity (dBm)');
colormap(jet);shading interp 


xlim([-40 40]);ylim([0 80]);zlim([-40 0]);
caxis([-40 -10]);view(-44,41)

h=colorbar;
title(h,'dB');
a = h.Position;
set(h,'Position',[a(1)+0.08 a(2)+0.23 0.01 0.5]);
set(gcf,'color',[1 1 1]);
set(gca,'FontSize',fontsz)
set(findall(gcf,'type','text'),'FontSize',fontsz);
set(get(gca,'YLabel'),'Rotation',-26);
set(get(gca,'XLabel'),'Rotation',26);

 % MIMO Raw Data Simulation
 MIMO_data(Na:N_Fast)=0;
for k=1:Na
    echo=TX_RF*0;
    for tn=1:length(target_range)
        range=target_range(tn);
        echo= echo+ target_rcs(tn)*exp(1i*pi*Kr*(t-2*range/c0).^2).*exp(1i*2*pi*Fc*(t-2*range/c0)) *exp(-1i*2*pi*(k-1)*delta_dis*sind(target_theta(tn))/lambda)/ (range^4);
    end
    Gain    =30+ Pt-30+Gt +Gr + 20*log10(lambda)+LNA_ADC_Gain-30*log10(4*pi)-L;
    Gain    =sqrt(10^(Gain/10)); %convert power gain to voltage gain
    echo   = echo * Gain;
    echo=echo+ noise_amp.*(randn(1,length(echo))+1i*randn(1,length(echo))); 
    Mixer_Output = conj(echo.*TX_Ref); %混频
    Mixer_Output = decimate(Mixer_Output, RF_fs/fs);
    RX_Base  = filter(Hd, Mixer_Output); 
    MIMO_data(k,:)=RX_Base;
end

rafData=fftshift(fft(MIMO_data,[],2),2);
rafData=rafData(:,1+N_sig/2:N_sig);

Ridx=dletaR*[0:N_Fast/2-1];
Na_fft=256;
AzNfft=Na_fft;
az0=linspace( -1,1,Na_fft);
azz=asind(az0);
data=fftshift(fft(rafData,AzNfft,1),1);
figure;
Xr=idR'*cosd(azz);
Yr=idR'*sind(azz);
tdata=20*log10(abs(data)/max(abs(data(:))+eps));

pcolor(Yr',Xr',tdata);
shading interp
axis;
xlim([-R_max   R_max]);
ylim([0    R_max]);
colormap(jet);
h=colorbar;
title(h,'dB')
xlabel('X (m)');
ylabel('Y (m)');
a = h.Position;
set(h,'Position',[a(1)+0.08 a(2)+0.05 0.01 0.6]);
set(gcf,'color',[1 1 1]);
set(gca,'FontSize',fontsz)
set(findall(gcf,'type','text'),'FontSize',fontsz);