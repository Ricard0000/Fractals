%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Basic Neural Networks for function fitting in 1 and 2-dimension.       
%This code uses y(x)=x on the interval [a,b]=[-1,1]                     
%And computes a 5 layered weight matrix to fit                          
%an output of y(x)=x^2=W_{N_layers+1}*sigmoid(W_{N_layers}*h_{N_layers})
%i.e. bias parameter is 0. (NOTE!!! This could be a problem if you are trying
%to use this code for your application!!)                       
%The 2-dimensional graph is also a quadratic.                           
%Maybe in the future I will include a non-zero bias in this code.                             
%This code shows two things:
%1) It is difficult to code up gradient descent from scratch
%and 2) I am good at Tensor Calculus.
%The code is written in Matlab, so it is easy to use, no                
%need to download any external modules. All you need to do              
%is copy and paste this code to a script file and run.                  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

format long

%%%%%%%%%%%%%%%%%%%%
%1-Dimensional code%
%%%%%%%%%%%%%%%%%%%%
N_layers=2;
N_layer_size=10;
N_iter=1000;
Nx=100;
a=-1;
b=1;
x=zeros(Nx,1);
output=zeros(Nx,1);
%This is the Domain%
for I=1:Nx
    x(I,1)=a+(b-a)/(Nx-1)*(I-1);
end
%This is what you are fitting to%
for I=1:Nx
    output(I,1)=x(I)^2;
end

y_out_1d=Neural_Net(Nx,N_layer_size,N_layers,N_iter,x,output);


figure(1)
plot(x,y_out_1d,'r','linewidth',2)
title('The fitted function $y(x)$','Interpreter','Latex','FontSize',18)

figure(2)
plot(x,abs(x.*x-y_out_1d),'r','linewidth',2)
title('Error: abs(Exact-NeuralNet)','Interpreter','Latex','FontSize',18)
clear


%%%%%%%%%%%%%%%%%%%%
%2-dimensional Code%
%%%%%%%%%%%%%%%%%%%%
N_layers=2;
N_layer_size=20;
N_iter=500;
Nx=16;
a=-1;
b=1;
Ny=16;
c=-1;
d=1;
for I=1:Nx
    x(I,1)=a+(b-a)/(Nx-1)*(I-1);
    y(I,1)=c+(d-c)/(Ny-1)*(I-1);
end
%This is what you are fitting to%
for I=1:Nx
    for J=1:Ny
        output(I,J)=1-x(I)^2-y(J)^2;
    end
end

%Reshaping 2-dimensional data to a 1-D vector%
N=Nx*Ny;
L=1;
xx=meshgrid(y,x);
for I=1:Nx
    for J=1:Ny
        output_2d(L,1)=output(I,J);
        x_2d(L,1)=xx(I,J);
        L=L+1;
    end
end

y_out_2d=Neural_Net(N,N_layer_size,N_layers,N_iter,x_2d,output_2d);

L=1;
for I=1:Nx
    for J=1:Ny
        y_out_2dd(I,J)=y_out_2d(L);
        L=L+1;
    end
end

figure(3)
mesh(y,x,y_out_2dd)
title('The fitted $z(x,y)$ in 2D','Interpreter','Latex','FontSize',18)

figure(4)
mesh(y,x,y_out_2dd-output)
title('Error: abs(Exact-NeuralNet) in 2D','Interpreter','Latex','FontSize',18)


%clear




%%%%%%%%%%%%%%%%%%%
%Modules down here%
%%%%%%%%%%%%%%%%%%%
function [y_out]=Neural_Net(Nx,N_layer_size,N_layers,N_iter,x,output)
%Initialization of input=h(:,1)%
    for I=1:Nx
        y_in(I,1)=x(I);
    end
%This is what you initialize the weights%
    W1=0.2*ones(N_layer_size,Nx);
    W=0.2*ones(N_layer_size,N_layer_size,N_layers-1);
    W2=0.2*ones(Nx,N_layer_size);
%Hiddenlayers denoted by h%
    h=zeros(N_layer_size,N_layers);
%Derivative of weights%
    dW1=zeros(N_layer_size,Nx);
    dW=zeros(N_layer_size,N_layer_size,N_layers-1);
    dW2=zeros(Nx,N_layer_size);
%Initial Caculation of hidden layers%
    h(:,1)=sigmoid(W1(:,:)*y_in);
    for I=1:N_layers-1
        h(:,I+1)=sigmoid(W(:,:,I)*h(:,I));
    end
    y_out=W2*h(:,N_layers);
%Initial Error%
    T1=(output-y_out);
    Error=0.5/Nx*(T1'*T1)
%Start of iterations%
    if( N_layers==1)
        K=1;
        while((K<N_iter)&&(Error>10^-16))
            h(:,1)=sigmoid(W1(:,:)*y_in);
            y_out=W2*h(:,1);
            T1=(output-y_out);
            Error=(0.5/Nx)*(T1'*T1)
            for I=1:Nx
                dW2(I,:)=-1/Nx*T1(I)*h(:,1)';
            end
            T2=W2*h(:,1);
            for I=1:N_layer_size
                dW1(I,:)=-1/Nx*(output-T2)'*W2(:,I)*(dsigmoid(W1(I,:)*y_in))*y_in';
            end
            W1=W1-dW1;
            W2=W2-dW2;
        K=K+1;
        K
        end
    elseif N_layers==2
        K=1;
        while((K<N_iter)&&(Error>10^-16))
            h(:,1)=sigmoid(W1(:,:)*y_in);
            for I=1:N_layers-1
                h(:,I+1)=sigmoid(W(:,:,I)*h(:,I));
            end
            y_out=W2*h(:,N_layers);
            T1=(output-y_out);
            Error=(0.5/Nx)*(T1'*T1)
            for I=1:Nx
                dW2(I,:)=-1/Nx*T1(I)*h(:,1)';
            end
            for I=1:N_layer_size
                T123=W2*dsigmoid(W(:,:,N_layers-1)*h(:,N_layers))*h(:,N_layers)';
                dW(I,:,N_layers-1)=-1/Nx*(output-T1)'*T123;
            end
            for I=1:N_layer_size
                T12=W2*(dsigmoid(W(:,:,N_layers-1)*h(:,N_layers)).*W(:,I,N_layers-1))*dsigmoid(W1(I,:)*y_in)*y_in';
                dW1(I,:)=-1/Nx*(output-T1)'*T12(:,:);
            end   
            W2=W2-dW2;
            W(:,:,N_layers-1)=W(:,:,N_layers-1)-dW(:,:,N_layers-1);
            W1=W1-dW1;
        K=K+1;
        K
        end
    else
        K=1;
        while((K<N_iter)&&(Error>10^-16))
            h(:,1)=sigmoid(W1(:,:)*y_in);
            for I=1:N_layers-1
                h(:,I+1)=sigmoid(W(:,:,I)*h(:,I));
            end
            y_out=W2*h(:,N_layers);
            T1=(output-y_out);
            Error=(0.5/Nx)*(T1'*T1)
            for I=1:Nx
                dW2(I,:)=-1/Nx*T1(I)*h(:,1)';
            end
            for I=1:N_layer_size
                T123=W2*dsigmoid(W(:,:,N_layers-1)*h(:,N_layers))*h(:,N_layers)';
                dW(I,:,N_layers-1)=-1/Nx*(output-T1)'*T123;
            end
            for J=1:N_layers-2
                for I=1:N_layer_size
                    T1234=W(:,:,N_layers-1-J)*dsigmoid(W(:,:,N_layers-1-J)*h(:,N_layers-J-1))*h(:,N_layers-J)';
                    dW(I,:,N_layers-1-J)=dW(I,:,N_layers-J)*T1234';
                end
            end
            for I=1:N_layer_size
                T12=W2*(dsigmoid(W(:,:,1)*h(:,1)).*W(:,I,1))*dsigmoid(W1(I,:)*y_in)*y_in';
                dW1(I,:)=-1/Nx*(output-T1)'*T12(:,:);
            end
            W2=W2-dW2;
            for I=1:N_layers-1
                W(:,:,N_layers-I)=W(:,:,N_layers-I)-dW(:,:,N_layers-I);
            end
            W1=W1-dW1;
        K=K+1;
        K
        end
    end
end
function [out]=sigmoid(x)
    out=1./(1+exp(-x));
end
function [out]=dsigmoid(x)
    out=exp(-x)./(1+exp(-x)).^2;
end