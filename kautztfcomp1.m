function kautztfcomp1(x)
n=12;
%x=[-0.8;-0.61;0.21;0.19];
%mod(n,2)=0;
n1=n/2;
%  n2=n1/2;
 for i=1:n1
     a1(i,1) = x(i,1);
     b1(i,1) = x(i+n1,1);
 end
% else 
%     disp('order is not valid');
%     return
% end


for i=1:n1
% a1(i)=input('enter the value  of real part ');
%  b1(i)=input('enter the value of img. part ');
%  if sqrt(a1(i,1).^2+b1(i,1).^2)>1, 
%      a1=a1
%      b1=b1
%      error('Unstable poles');
% 
%  else
    beta(i,1) = complex(a1(i,1),b1(i,1));
    beta1(i,1) = conj(beta(i,1));
   
%  end
end


for i=1:n1
 h1(i)= -(beta(i,1)+beta1(i,1));
 h2(i)=beta(i,1).*beta1(i,1);
end
disp(h1);
disp(h2);

% get the matrices of A1
A1=eye(n1);
for i=1:n1
    for j=1:n1
        if i==j
            A1(i,j)=-(h1(i));
        elseif i>j
            A1(i,j)=h1(j).*(1-h2(j));
        elseif i-j>1
            A1(i,j)=A1(i-1,j).*h2(i-1);
        else
            A1(i,j)=0;
        end
    end
end
disp(A1);
% get the matrices of A2
h22=h2.^2;
A2=eye(n1);
for i=1:n1
    for j=1:n1
        if i==j
            A2(i,j)=-(h2(i));
        elseif i>j
            A2(i,j)=(1-h22(i));
        elseif i-j>1
            A2(i,j)=A2(i-1,j).*h2(i-1);
        else
            A2(i,j)=0;
        end
    end
end
disp(A2); 
% to get the vectors of B
B1=eye(n1,1);
for i=1:n1
    j=1;
    if i-j==1
        B1(i,1)=h2(i);
    elseif i-j>1
        B1(i,1)=B1(i-1,j).*h2(i-1);
    end
end
% disp(B1);
% to get the even and odd states
% % x=repmat([1;2],[n/2,n/2]);
% % k=1;
% % j=1;
% % for i=1:n
% %     if mod(i,2)==0
% %         xeven(j)=x(i);
% %         j=j+1;
% %     else
% %         xodd(k)=x(i);
% %         k=k+1;
% %     end
% % end 
%    



% % generating an underdamped system
% wn=[1 2];
% zeta=[0.3 0.4];
% for i=1:n1
% %     wn(i)=input('enter the values of wn ');
% %     zeta(i)=input('enter the values of zeta ');
%     if zeta(i)>0.75
%         error('not valid')
%     else 
%        G(i)=tf(wn(i).^2,[1 2*zeta(i)*wn(i) wn(i).^2]);
%     end 
% end
% for i=1:n1
%    if i==1 
% %        G1(i)=series(G(i),G(i+1));
% G1(i)=G(i);
%    elseif i>1 && i<=n1
%        G1(i)=series(G1(i-1),G(i));
%        
%        
%        
%     
%     end
%     
% end
% G2=G1(n1);
% G3=c2d(G2,0.1,'zoh');
% figure(1);
% step(G2);
% % r=rand(1,50);
% % ww=1;
% % for i=1:length(r)
% % for j=1:18
% % D(1,ww)=r(i);
% % ww=ww+1;
% % end
% % end
% load PsedoIP.mat;
% % D = 
% 
% % D = ones(2500,1);
% 
% T=[0:length(D)-1].*0.1;
% %figure,plot(D)
% B=lsim(G3,D,T);
% figure(3);
% plot(T,B);
load TF_ip.mat;
load TF_op.mat;
load TF_up.mat;
%load TF_up1.mat;
%evaluate system states
% xodd=zeros(2,1);
xodd1=[];
xeven1=[];
xodd2=[];
xodd1=zeros(n1,2);
%xeven1=zeros(n1,1);

% get the kautz model states
for k=1:length(D)
     if k>=3
    xodd1(:,k)=A1*xodd1(:,k-1)+A2*xodd1(:,k-2)+B1*D(k-1);
    xeven1(:,k)=xodd1(:,k-1);
    end
end
% cc=2;dd=2;
% for k=1:length(D)
%     if mod(k,2)==0
%         xeven1(:,cc)=xodd1(:,k);
%         cc=cc+1;
%     else
%         xodd2(:,cc)=xodd1(:,k);
%         dd=dd+1;
%     end
%         
% end

% for i=1:n1
%     for j=1:n1
%     if A1(i,j)>=0 
%         A11(i,j)=A1(i,j);
%         
%     elseif A1(i,j)<0 
%         A12(i,j)=A1(i,j);
%         
%     end
%     end
% end
% A12=abs(A12);
% 
% for i=1:n1
%     for j=1:n1
%     if A2(i,j)>=0 
%         A21(i,j)=A2(i,j);
%         
%     else
%         A22(i,j)=A2(i,j);
%         
%     end
%     end
% end
% A22=abs(A22);

% get c matrices
A3=A1.*A1;
A4=A2.*A2;
H=h1.^2;
H1=h2.^2;
for i=1:n1
    c1(:,i)=sqrt(abs((1-H(:,i)+3*H1(:,i))*(1-h2(:,i))./(1+A3(:,i))*(1+h2(:,i)).*(2*A1(:,i)*h1(:,i))));
    c2(:,i)=sqrt(abs((1-H(:,i)+3*H1(:,i))*(1-h2(:,i))./(1+A4(:,i))*(1+h2(:,i)).*(2*A2(:,i)*h1(:,i))));
end

% for i=1:n1
%     c1=sqrt(abs((1-H(i)+3*H1(i))*(1-h2(i))./(1+A3(:,i))*(1+h2(i)).*(2*A1(:,i)*h1(i))));
%     c2=sqrt(abs((1-H(i)+3*H1(i))*(1-h2(i))./(1+A4(i))*(1+h2(i)).*(2*A2(:,i)*h1(i))));
% end
  
for i=1:length(xeven1)
    phi(:,i)=c1*(xeven1(:,i)-A1*xodd1(:,i));
    phi1(:,i)=c2*(xeven1(:,i)-A2*xodd1(:,i));
end


        

Z=[];
cc=1;
%n2=n1-1;
[m,n]=size(phi)
for k=1:size(xeven1,1)
        Z(:,cc)=phi(k,:)';
        cc=cc+1;
        Z(:,cc)=phi1(k,:)';
        cc=cc+1;
    end
save zmat Z;
% disp(phi);
% disp(phi1);

% Matrix concatenation
%zz = Z;
% Z=[phi phi1];
% save zmat Z
% Nrm_Z=norm(Z,1)

%Theta = theta;
% disp(Z),
% figure(5),plot(Z)
% pause,
% close(5)
% least square approximation
theta=inv(Z'*Z)*Z'*B;                 % B =underdamped+arbitrary step signal
% if isnan(theta)
%     theta = Theta;
%     Z = zz;
%     disp(x)
%     pause;
% end
% transfer function for kautz
Y=Z*theta;
for i=1:length(D)
    if i>1
    Y1(i)=Y(i)-Y(i-1);
    end
end
save  Y1;

% Y = sum([phi;phi1],1);
%figure(3),plot(B);
%figure(3),hold on,plot(Y,'r');
hold off;
legend('system','model');
N=720;
Fs=1;
tt=0:1/Fs:(N-1)/Fs;
ay=hilbert(Y);
%figure,plot(tt,real(ay),tt,imag(ay));
%title('hilbert version of kautz output');
MAY=abs(ay);
PAY=angle(ay);
instfreq=Fs/2*pi*diff(unwrap(angle(ay)));
%figure,plot(tt(2:end),instfreq);
%title('instfrequency of kautz output');
ay1=hilbert(B);
%figure,plot(tt,real(ay1),tt,imag(ay1));
%title('hilbert version of system output');
MAY1=abs(ay1);
PAY1=angle(ay1);
instfreq1=Fs/2*pi*diff(unwrap(angle(ay1)));
%figure,plot(tt(2:end),instfreq1);
%title('instfrequency of system output');
plot(tt(2:end),instfreq,'r'),hold on,plot(tt(2:end),instfreq1,'g')

%figure,plot(Y,'g-'),set(gca,'Ydir','normal'),text(locs1+.2,pks,num2str((1:numel(pks))'));



