function [err]=kautzobj(xx)
% clc;
% n=10;
n=2;
%x=[-0.8;-0.65;0.24;0.21];
%mod(n,2)=0;
n1=n/2;
load xx;
%  n2=n1/2;
% load state
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
            A1(i,j)=-(beta(i,1)+beta1(i,1));
        elseif i>j
            A1(i,j)=(beta(j,1)+beta1(j,1))*(1-beta(j,1)*beta1(j,1));
        elseif i-j>1
            A1(i,j)=A1(i-1,j).*(beta(i-1,1)*beta1(i-1,1));
        else
            A1(i,j)=0;
        end
    end
end
% disp(A1);
% get the matrices of A2
% h22=h2.^2;
A2=eye(n1);
for i=1:n1
    for j=1:n1
        if i==j
            A2(i,j)=-(beta(i,1)*beta1(i,1));
        elseif i>j
            A2(i,j)=1-(beta(i,1)*beta1(i,1)).^2;
        elseif i-j>1
            A2(i,j)=A2(i-1,j).*(beta(i-1,1)*beta1(i-1,1));
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
        B1(i,1)=beta(i,1).*beta1(i,1);;
    elseif i-j>1
        B1(i,1)=B1(i-1,j).*(beta(i-1,1).*beta1(i-1,1));
    end
end
A3=diag(A1);
A4=diag(A2);

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
% load TF_ip.mat;
% load TF_op.mat;
% D=ones(9000,1);
% D=xlsread('data11.xlsx','E3:E9002');
% BB=xlsread('data11.xlsx','D3:D9002');
load x
load y
% load ip;
% load op;
% load TF_up.mat;
%load TF_up1.mat;

%evaluate system states
% xodd=zeros(2,1);
xodd1=[];
xeven1=[];
xodd2=[];
% xodd1=zeros(n1,2);
%xeven1=zeros(n1,1);

% get the kautz model states

for k=1:length(input)
     if k>=3
    xodd1(:,k)=A1*xodd1(:,k-1)+A2*xodd1(:,k-2)+B1*input(k-1)';
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
A3=A1*A1;
A4=A2*A2;
H=h1.^2;
H1=h2.^2;
 for i=1:n1;
   
    c1(:,i)=sqrt(abs(((1-H(:,i)+3*H1(:,i))*(1-h2(:,i)))./(1+A3(:,i))*(1+h2(:,i)).*(2.*A1(:,i)*h1(:,i))));
    c2(:,i)=sqrt(abs(((1-H(:,i)+3*H1(:,i))*(1-h2(:,i)))./(1+A4(:,i))*(1+h2(:,i)).*(2.*A2(:,i)*h1(:,i))));
 end
% % for i=1:n1
% %     c1=sqrt(abs((1-H(i)+3*H1(i))*(1-h2(i))./(1+A3(:,i))*(1+h2(i)).*(2*A1(:,i)*h1(i))));
% %     c2=sqrt(abs((1-H(i)+3*H1(i))*(1-h2(i))./(1+A4(i))*(1+h2(i)).*(2*A2(:,i)*h1(i))));
% 
% % end
% 
% %  C=[c1 c2];
% %  Y=C'*xodd1;
% % figure, plot(Y');
%  load TF_op B;
%  for i=1:length(D)
%      if i>1
%          D1(i)=D(i)-D(i-1);
%      end
%  end
% 
%   for i=1:length(D)
%       if i>1
%       dxodd1(:,i)=xodd1(:,i)-xodd1(:,i-1);
%       end
%   end
%   
%   for k=1:length(D)
%      if k>=3
%     ddxodd2(:,k)=A1*dxodd1(:,k-1)+A2*dxodd1(:,k-2)+B1*D(k-1);
%     ddxeven1(:,k)=ddxodd2(:,k-1);
%     end
%   end

  for i=1:length(xeven1)
    phi2(:,i)=c1*(xeven1(:,i))-A1*xodd1(:,i);
    phi3(:,i)=c2*(xeven1(:,i))-A2*xodd1(:,i);
    
  end
  
  [m,n3]=size(phi2);
  cc=1;
for k=1:size(xeven1,1)
        Z(:,cc)=phi2(k,:)';
        cc=cc+1;
        Z(:,cc)=phi3(k,:)';
        cc=cc+1;
    end
In_Z1=inv(Z'*Z);
[mm,nn]=size(In_Z1)
theta=inv(Z'*Z)*Z'*s1;  % least square 

Y=Z*theta;
%Y1=Z*theta;
% %Y1=Y';
% for i=1:length(D)
%     if i>1
%     Y1(i)=Y(i)-Y(i-1);
%     end
% end
% save Y1;
%Y2=Y-Y1;
%Y1=Z*theta;
% save output Y;
% load TF_up T;
% load g5;

% figure,plot(Y1,'r')
% 
% hold on,plot(B);
%   
%          
% % plot(B,Y);
% 
% for i=1:length(xeven1)
%     phi(:,i)=c1*(xeven1(:,i))-(A1*xodd1(:,i));
%     phi1(:,i)=c2*(xeven1(:,i))-(A2*xodd1(:,i));
% end
% Z=[];
% cc=1;
% %n2=n1-1;
% [m,n]=size(phi)
% for k=1:size(xeven1,1)
%         Z(:,cc)=phi(k,:)';
%         cc=cc+1;
%         Z(:,cc)=phi1(k,:)';
%         cc=cc+1;
%     end
%         
% 
%         
% 
% % % to get regression vector
% % % Z = [];
% % % s=1;
% % % for i=1:length(D)
% % % %    phi(i,:)= [c1*(xeven1(:,i)-A1*xodd1(:,i))];
% % %    Z([s,s+n1],i) = phi(:,i);
% % %    s = s+1;
% % %    phi1(i,:)=[c2*(xeven1(:,i)-A2*xodd1(:,i))];
% % %    Z([s,s+n1],i) = phi1(:,i);
% % %    s=1;
% % % end
% % % save zmat Z;
% % % disp(phi);
% % % disp(phi1);
% 
% % Matrix concatenation
% %zz = Z;
% % Z=[phi phi1];
% % save zmat Z
% % Nrm_Z=norm(Z,1)
% 
% %Theta = theta;
% % disp(Z),
% % figure(5),plot(Z)
% % pause,
% % close(5)
% % least square approximation
% x
% In_Z=inv(Z'*Z);
% [mm,nn]=size(In_Z)
% theta=inv(Z'*Z)*Z'*B                 % B =underdamped+arbitrary step signal
% % if isnan(theta)
% %     theta = Theta;
% %     Z = zz;
% %     disp(x)
% %     pause;
% % end
% % transfer function for kautz
% Y=Z*theta;
% %Y1=upsample(Y,3)
% save op;
% % Y = sum([phi;phi1],1);
% Hilbert view of output signal
% A11=[A1 A2; eye(n1,n1) zeros(n1,n1)];
% B11=[B1;B1];
% H11=ctrb(A11,B11);
% H122=obsv(A11,theta');
% % calculate the teoplitz matrices using markov parameters
% G6=zeros(n,n);
% for i=1:n
%     for j=1:n
%      
% if i==j
%             G6(i,j)=theta'*B11;
% end
% if i-j==1
%      G6(i,j)=theta'*A11*B11;
% end
% if i-j==2
%     G6(i,j)=theta'*A11.^2*B11;
% end
%     for n2=3:n
%          if i-j==n2
%     G6(i,j)=theta'*A11.^((n2+1)-1)*B11;
%          end
%     end
%         
%     end
% end
% disp(G6);
% % % G6(1,:)=[];
% % % G6(:,n)=[];
% [U,S,V]=svd(G6);
% A111=(-sqrt(S))*U'*G6*V*(-sqrt(S));

% G7=zeros(n,n);
% for i=1:n
%     for j=1:n
%         if i==j
%         G7(i,j)= theta'*A11*B11;
%         end
%         if i-j==1
%             G7(i,j)=theta'*A11.^2*B11;
%         end
%         if i-j==-1
%             G7(i,j)=theta'*B11;
%         end
%         if i-j==2
%             G7(i,j)=theta'*A11.^3*B11;
%         end
%         for n4=3:n
%             if i-j==n4
%                 G7(i,j)=theta'*A11.^((n4+2)-1)*B11;
%             end
%         end
%     end
% end
% [U1,S1,V1]=svd(G7);
% 
% G8=zeros(n,n);
% for i=1:n
%     for j=1:n
%         if i==1 && j==1
%             G8(i,j)=theta'*B11;
%         end
%         if i==2 && j==2
%             G8(i,j)=theta'*A11.^2*B11;
%         end
%          for n1=3:n+1
%         if i==n1 && j==n1 ;
%           k=2*(n1-1);
%          G8(i,j)=theta'*(A11.^k)*B11;
%         end
%          end   
%         for n6=2:n
%                 if i==n6 && j==1
%                 G8(i,1)=theta'*A11.^(n6-1)*B11;
%                 end
%                 if  j>1 && i-j==n6
%                     k=j;
%                     G8(i,j)=theta'*A11.^k*B11;
%                 end
% 
%         end
%             for n7=2:n
%                 if j==n7 && i==1
%                     G8(1,j)=theta'*A11.^n7*B11;
%                 end
%                 if i>1 && j-i==n7
%                     k=i;
%                 G8(i,j)=theta'*A11.^k*B11;
%                 end
%             end
%                 
%     end
% end
% 
% 
% 
% G9=zeros(n,n);
% for i=1:n
%     for j=1:n
%         if i==1 && j==1
%             G9(i,j)=theta'*A11*B11;
%         end
%         if i==2 && j==2
%             G9(i,j)=theta'*A11.^3*B11;
%         end
%          for n1=3:n
%         if i==n1 && j==n1 ;
%           k=(2*n1)-1;
%          G9(i,j)=theta'*(A11.^k)*B11;
%         end
%          end      
%         for n6=1:n
%                 if i-j==n6
%                 G9(i,j)=theta'*A11.^(n6+1)*B11;
%                 end
%         end
%             for n7=1:n
%                 if j-i==n7
%                 G9(i,j)=theta'*A11.^(n7+1)*B11;
%                 end
%             end
%     end
% end
% 
%              
%                  
%             
% 
% %disp(G6);
% Manual SDP model of x^2 + y^2 <= 1

 % Hilbert transfrom of the kautz system           
% N=720;
% Fs=1;
% tt=0:1/Fs:(N-1)/Fs;
% ay=hilbert(Y);
% MAY=abs(ay);
% PAY=angle(ay);
% instfreq=Fs/2*pi*diff(unwrap(angle(ay)));
% ay1=hilbert(B);
% MAY1=abs(ay1);
% PAY1=angle(ay1);
% instfreq1=Fs/2*pi*diff(unwrap(angle(ay1)));
% save op1;
% generate input and output vectors
% Dn=D';    % input sequence
% Dn=[Dn;zeros(n-1,1)];
% i=1:(n+80)-1;
% JI=[Dn delayseq(Dn,-i)];
% JI( ~any(JI,2), : ) = [];  %rows
% JI( :, ~any(JI,1) ) = [];  %columns 
% Yn=[Y;zeros(9,1)]; %output sequence
% JO=[Yn delayseq(Yn,-i)];
% JO( ~any(JO,2), : ) = [];  %rows
% JO( :, ~any(JO,1) ) = [];  %columns 
% load ddd;
% [vv,ww]=ss2tf(A11,B11,theta',0);
% sys=tf(vv,ww,ddd);
% [mag,phase]=bode(sys);
% zz=eig(A1);
% zz1=eig(A2);
% zz2=[zz;zz1];
% A11=[A1 A2; eye(n1,n1) zeros(n1,n1)]; %concatenation of signal
% % B11=[B1;B1];
% B11=[1;0];
% P=sdpvar(2,2);
% Q=sdpvar(2,2);
% R=sdpvar(1,1);
% xest=sdpvar(2,1);
% objective=norm((s1-Y),1);
% for i=1:1000
% Pi=sdpvar(2,2);
% M1=-A11*Pi-Pi*A11+Pi*B11*1*B11'*pi-Q;
% % xest=A11*xest+B11*D(i)-k'*(theta'*xest-c_1*xest);
% end
% const=[M1<0,-P>0,-Q>0,-R>=0,zz2.^2-1];
% optimize(const,objective);
% p=double(Pi);
% q=double(Q);
% r=double(R);
% k=inv(r)*B11'*p;


save op2;
err=norm((s1 - Y),1)