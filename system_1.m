clc;
clear all;
close all;
% %  n=2;
% % n1=n/2;
% % f= 1818;   % frequency in hertz
% % f1=f/1000; %frequency in kilohertz
fs=100000000;
% % wn1=2*pi*f;
% % zeta=0.55 ;
% % for i=1:n1
% % %     wn(i)=input('enter the values of wn ');
% % %     zeta(i)=input('enter the values of zeta ');
% %     if zeta(i)>0.75
% %         error('not valid')
% %     else 
% %        G(i)=tf(wn1(i).^2,[1 2*zeta(i)*wn1(i) wn1(i).^2]);
% %     end 
% % end
% % zeta=0.41;
% % g1=tf(wn1.^2,[1 2*zeta*wn1 wn1.^2]);
% %  zeta1=0.13;
% %         g2=tf(wn1.^2,[1 2*zeta1*wn1 wn1.^2]);
% %         
% %          zeta1=0.18;
% %         g3=tf(wn1.^2,[1 2*zeta1*wn1 wn1.^2]);
% %     
% % for i=1:n1
% %    if i==1 
% % %        G1(i)=series(G(i),G(i+1));
% % G1(i)=G(i);
% %    elseif i>1 && i<=n1
% %        G1(i)=series(G1(i-1),G(i));
% %    end
% % end
% % G2=G1(n1);
% % G3=c2d(G2,0.001,'zoh');
% % G4=ss(G3);
% % for i=1:n1
% %    if i==1 
% % %        G1(i)=series(G(i),G(i+1));
% % g1(i)=g(i);
% %    elseif i>1 && i<=n1
% %        g1(i)=series(g1(i-1),g(i));
% %    end
% % end
% % 
% % for i=1:n1
% %    if i==1 
% % %        G1(i)=series(G(i),G(i+1));
% % g2(i)=g(i);
% %    elseif i>1 && i<=n1
% %        g2(i)=series(g2(i-1),g(i));
% %    end
% % end
% % 
% % 
% % g3=g1(n1);
% % g4=g2(n1);
% % fs=10*f;
% % g5=c2d(g1,1/fs,'zoh');
% % save g5;
% % g6=c2d(g2,0.00005,'zoh');
% % g7=c2d(g3,0.00005,'zoh');
% % 
% % g5=ss(g3);
% % g6=ss(g4)
% % save TF G3;
% % pole(G3);
% % 
% % save pole G3;
% % figure(1);
% % step(G2);
% % r=rand(1,8);
% % ww=1;
% % for i=1:length(r)
% %     for j=1:100
% %         D(1,ww)=r(i);
% %         ww=ww+1;
% %     end
% % end
% % fs=1000;
% % load freq_est
% % tt=0:1/fs:(1.5)-1/fs;
% % D=chirp(tt,f,1,f/2);
% % D=sin(2*pi*f*tt);
% % plot(tt,D)
% D=xlsread('data11.xlsx',1,'E3:E9002');
D=xlsread('data11.xlsx',1,'E3:E9002');
% D1=D';
% t=0:0.01:1;
% D=2.*(t>=0);
% t=-0.09:0.01:1;
% D=2.*(t==0);
% plot(t,D);
% 

% %D4 = ones(500,1);
% %save step D4;
% %D=D.*1000;
% %D=D.*0.001
% t=1:720;
% D=t>=0;
% T=[0:(length(D))-1].*0.00006;
% T=[0:0.1:2*pi].*0.0007;
% plot(D)
% BB=lsim(g5,D,tt);
% BB1=lsim(g6,D,tt);
% BB2=lsim(g7,D,tt);
figure,plot(D)
% figure,plot(BB),hold on,plot(D)%hold on,plot( tt,BB1,'r'),hold on,plot(tt,BB2,'g');
% figure,plot(tt,BB,fn'r'),hold on,plot(tt,D,'b')
% figure,stem(tt,BB) %hold on,stem(tt,BB1)
% [pks,locs]=findpeaks(BB);
% [pks1,locs1]=findpeaks(BB1);
% [pks2,locs2]=findpeaks(BB2);
% zeta_est=1/(2*(max(pks-pks1)+max(pks1-pks2)));
%  [pxx,f]=pwelch(BB,2000,1000,5000,20000);
%  figure,plot(f,10*log10(pxx));
%   [pxx,f]=pwelch(BB1,2000,1000,5000,20000);
%   figure,plot(f,10*log10(pxx));
%   [pxx,f]=pwelch(BB2,2000,1000,5000,20000);
%   figure,plot(f,10*log10(pxx));
% xlim([0 0.1]);
%B=upsample(B,3);
Nfft=1024;
[Pxx,f] = pwelch(D,gausswin(Nfft),Nfft/2,Nfft,5000000);
% [pxx1,f1]=pwelch(D,length(D),4096,[],fs)
% [pxx1,f1]=pwelch(D,length(D),4096,[],fs);
figure,plot(f,10*log10(Pxx))
[~,loc] = max(Pxx);
FREQ_ESTIMATE = f(loc);
title(['Frequency estimate = ',num2str(FREQ_ESTIMATE),' Hz']);
% [pxx2,f2]=pwelch(BB,length(BB),1024,[],fs);
% figure,plot(f2,10*log10(pxx2))
% [pks,locs]=findpeaks(pxx2);
% save TF_op BB ;
% save TF_ip D;
xn=hilbert(D);
figure,plot(real(xn),imag(xn));
MAY1=abs(xn);
PAY1=angle(xn);
[Pxx1,f1] = pwelch(xn,gausswin(Nfft),Nfft/2,Nfft,5000000)
figure,plot(f1,10*log10(Pxx1))
% fs1=max(f3);
% freq_est=fs1/(10);
save freq_est;
% [NUM RAW TXT]=xlsread('data','sheet1','A1:A100');
% ddata=iddata(BB,D',1/fs);
% ggg=tfest(ddata,2,0);
% nnn=c2d(ggg,1/fs,'zoh');
% xx=damp(BB);
% xx1=max(xx);
% zeta_est=1/(2*xx1);
% aa=-(zeta*wn1);
% bb=wn1*sqrt(1-(zeta).^2);
% cc=-(wn1*sqrt(1-(zeta).^2));
freq_est=FREQ_ESTIMATE;
zeta_est=max(damp(D));
% sam_time=1/fs;
omega_est=2*pi*freq_est;
aaaa=-(zeta_est*omega_est);
bbbb=aaaa-(j*freq_est*sqrt(1-(zeta_est).^2));
cccc=aaaa+(j*freq_est*sqrt(1-(zeta_est).^2));
% cccc=-(omega_est*sqrt(1-(zeta_est).^2));
% x1=complex(aaaa,bbbb);
% x2=complex(aaaa,cccc);
X=exp(bbbb*0.0000002);
    X1=exp(cccc*0.0000002);
aaa=real(X);
bbb=imag(X);
ccc=imag(X1);
x=[aaa;bbb];
x2=[aaa;ccc];
save x;
% save x2;
% save TF_up T;
%save TF_up1 D1;

% xx=damp(BB);
% xx1=max(xx);
% zeta_est=1/(2*max(xx));
N=length(tt);
for i=1:N-1
    pay1(i)=(PAY1(i+1)-PAY1(i))/(tt(i+1)-tt(i));
end
Dn=D';    % input sequence
% Dn=[Dn;zeros(n-1,1)];
i=1:n;
% D1=Dn(i);
JI=[Dn(i) delayseq(Dn(i),-i)];   % input hankel matrix
% JI( ~any(JI,2), : ) = [];  %rows
JI( :, ~any(JI,1) ) = [];  %columns 
% Bn=[BB;zeros(n-1,1)]; %output sequence
% B1=BB(i);
JO=[BB(i+1) delayseq(BB(i+1),-i)];    % output hankel matrix\
% JO( ~any(JO,2), : ) = [];  %rows
JO( :, ~any(JO,1) ) = [];  %discard columns 
j=(n+1):2*n;
D2=Dn(j);
JII=[D2 delayseq(D2,-i)];
JII( :, ~any(JII,1) ) = [];
B2=BB(j);
JOO=[B2 delayseq(B2,-i)];
JOO( :, ~any(JOO,1) ) = [];
n2=n*2;
m1=zeros(n2,n);
% % m2=[JOO;JII];
% % jj=1;
% % kk=1;
% for i=1:n
%  if  mod(m1(i,:),2)==1
%         m1(i,:)=JO(i,:);
% %         jj=jj+1;
%         
%  elseif mod(m1(i,:),2)==0
%         m1(i,:)=JI(i,:);
% %         jj=jj++1;
%  end
% end
% M = zeros(n,n);

% m1=[JI(1:2:end,:),JO(2:2:end,:)];
% m1(1:2:end,:)=JI([1:end],:);
% m1(2:2:end,:)=JO([1:end],:);

% m2=zeros(n2,n);
% m2(1:2:end,:)=JII([1:end],:);
% m2(2:2:end,:)=JOO([1:end],:);
% 
% m3=[m1;m2];
% 
% % m1([1:n],:)=JI([1:n],:);
% % m1(n,:)=JO([1:n],:);
% % 
% % m2=zeros(n2,n);
% % m2([1:n],:)=JII([1:n],:);
% % m2(n,:)=JOO([1:n],:);
% % m3=[m1;m2];
% 
% [U,S,V]=svd(m3);    % svd calculation of henkal matrix
% % U1=U([1:20],[1:30]);
% % U2=U([1:20],[31:40]);
% % U3=U([21:40],[1:30]);
% % U4=U([21:40],[31:40]);
% % S1=S([1:30],[1:10]);
% % Sn=horzcat(S,zeros(40,20));
% % Sn=Sn([1:30],[1:30]);
% % Ut=U2'*U1*Sn;
% % [U5,S5,V5]=svd(Ut);
% % X=U5*U2'*m3([1:20],:);
% % X1=U5*U2'*m3([3:22],:);
% % U11=m3([21:21],:);
% % Y11=m3([22:22],:);
% % Y12=[X1;Y11];
% % X12=[X;U11];
% % H=inv(X12'*X12)*X12'*Y12;
% % % H=X2*Y12;
% % SS=diag(S);
% % SS1=SS.^2;
% % save SS1;
% % SS2=diag(S5);
% % SS3=SS2.^2;
% % save SS3;
% % figure,plot(SS3,'-r');
% sys=n4sid([D';BB],2);
% lamda=0.99;
% N=20;
%initialize p-matrix
% delta=1e2;
% P=delta*eye(n);
% w=zeros(n,1);
% for n1=n:N
%     u=D(n1:-1:n1-n+1);
%     BB1=BB';
%     u1=BB(n1:-1:n1-n+1);
%     u11=[u';u1'];
%     phi=u11*P;
%     k=phi'./(lamda+(phi*u1));
%     BB(n1)=w'*u11;
% 	P=(P-k*phi)/lamda ;
% end
% 
% 
save ddd
% So = fft(BB); % This statement computes Fourier transform of x
% n = length(So); % length(x) gives the array length of signal x
% c = (-1 * fs) / 2:fs / n:fs / 2 - fs / n; % It generates the frequency series to plot X in frequency domain
% subplot(6, 1, 1),plot(tt,BB); % This subplot shows the signal x vs. time series t
% subplot(6, 1 ,2),plot(c,fftshift(abs(So))); % This subplot shows the Fourier spectrum of x with zero frequency component shifted to center
% subplot(6, 1, 3),plot(c,phase((So))); % This subplot shows the phase distribution of X (Fourier transform of x)
% subplot(6,1 ,4),plot(c,real(So)); % This subplot shows the real component of X spectrum







