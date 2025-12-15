% ================================================
Fs = 60e6;
Tw = 20e-6;
t = 1/Fs:1/Fs:Tw;
fc = 30e6;
v = 20;
c = 3e8;
lambda = c/fc;
fd = 2*v/lambda;

Tx = exp(1j*2*pi*fc*t);%发射信号
Rx = exp(1j*2*pi*(fc+fd)*t);%接收信号
afterMixer = Rx.*conj(Tx);%混频

figure(1);
X_fft = fftshift(fft(afterMixer));

deltaFreq = Fs/length(Tx);
xaxis = (-Fs/2:deltaFreq:Fs/2-deltaFreq)*lambda/2;
plot(xaxis,abs(X_fft));%绘图
title('多普勒测速')%添加标题
xlabel('速度(m/s)')%添加x轴注释






