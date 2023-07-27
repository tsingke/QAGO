% Quadruple parameter adaptation growth optimizer with integrated distribution, confrontation, and balance features for optimization
% School of Information Science and Engineering, Shandong Normal University, Jinan
% Corresponding author: Qingke Zhang
% Email:tsingke@sdnu.edu.cn
% Note: 1.The CEC 2017 file has been recompiled to return fitness value errors.
%       2.This version of GO uses the maximum number of evaluations (MaxFEs) as its termination criterion.
%       3.Need to load input_data_17 file for shifting, rotating, etc

clc;
clear;
close all;
format short e
rand('state', sum(100*clock));
% Parameter setting
ub=100;                 % Upper bound
lb=-100;                % Lower bound
N=40;                   % Population size (Too large value will slow down the convergence speed)
D=10;                   % Problem dimension (dimension=10/30/50/100)
MaxFEs=D*10000;         % Maximum number of evaluations MaxFEs=dimension*10000
Func=@cec17_func;       % Objective function set
FuncId=[];              % Function number

for FuncId=1:30
    fprintf("\n*******Current test function No.: F%d*******\n",FuncId);
    pause(2);
    [gbestX,gbestfitness,gbesthistory]= QAGO(N,D,ub,lb,MaxFEs,Func,FuncId);
    figure
    semilogy(1:MaxFEs,gbesthistory,'Linewidth',3);
    title(['F',num2str(FuncId),'  Optimization History'])
end