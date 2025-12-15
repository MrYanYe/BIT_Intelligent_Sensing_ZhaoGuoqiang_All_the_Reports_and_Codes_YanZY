%短时傅里叶变换展示
fs=2^10;    %采样频率 fs=65536hz
dt=1/fs;    %时间精度
timestart=-4;
timeend=4;
t=(0:(timeend-timestart)/dt-1)*dt+timestart;
L=length(t);

%设置信号
z=sin(2*pi*5.*t).*(t<-2)+sin(2*pi*10.*t).*(t>=-2&t<0)+...
    sin(2*pi*20.*t).*(t>=0&t<2)+sin(2*pi*40.*t).*(t>=2);
z2=wextend(1,'sym',z,round(length(z)/2));%镜像延拓

wlen=512;%设置窗口长度。窗口越长时间分辨率越差，频率分辨率越好。
hop=1;%每次平移的步长，最小为1。越小图像时间精度越好，但计算量大。
z2=wkeep1(z2,L+1*wlen);%中间截断


%做短时傅里叶
h=hamming(wlen);%设置海明窗的窗长
f=1:0.5:60;%设置频率刻度

[tfr2,f,t2]=spectrogram(z2,h,wlen-hop,f,fs);
tfr2=tfr2*2/wlen*2;
figure
imagesc(t2+timestart-wlen/fs/2,f,abs(tfr2))
