% Mandelbrot set generator.
%{c|z^2+c does not diverge starting at z=0}

L=1.25;
N=2000;
M=4000;
for I=1:N
    x(I)=-1.25+2.5/(N-1)*(I-1);
end
for I=1:M
    y(I)=-2+3/(M-1)*(I-1);
end

%z=x+1i*y;
Max=25;

%A=zeros(M,N);
A=ones(M,N);
for I=1:M
    for J=1:N
        II=1;
        c=y(I)+1i*x(J);
        z=0;
        conv=L/2;
%        while (((conv >= 10^(-15)) || (abs(z)>L)) && II<Max)
%        temp=z^2+c;
%        conv=abs(temp-z);
%        z=temp;
        for K=1:Max
            temp(K)=z^2+c;
            z=temp(K);
        end
        if (abs(z(:))<2)
            A(I,J)=-max(abs(z(:)));
        end
        
        
        %II=II+1;
        %if (abs(z)<L && conv < 10^(-15))
        %    A(I,J)=1;
        %end
        %end
    end
    I
end

figure(1)
%pcolor(x,y,A);
pcolor(A);
%surf(x,y,A);
shading interp
axis off
colormap(gray)
%figure(2)
%mesh(x,y,A)
clear
clc

