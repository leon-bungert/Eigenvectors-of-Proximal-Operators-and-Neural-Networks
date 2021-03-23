%---------------------------------------------------------------------
% This script computes an eigenvector of the FFDnet denoising neural
% network using the nonlinear power method from: 
% 
% Leon Bungert, Ester Hait-Fraenkel, Nicolas Papadakis, and Guy Gilboa. 
% "Nonlinear Power Method for Computing Eigenvectors of Proximal Operators 
% and Neural Networks." arXiv preprint arXiv:2003.04595 (2020).
%
% Please refer to this paper when you use the code.
% 
% authors: Leon Bungert <leon.bungert@fau.de>, 
% Ester Hait-Fraenkel <etyhait@campus.technion.ac.il>
% date: 23.03.2021
%---------------------------------------------------------------------

addpath(genpath('FFDnet_code'));
clear all;
close all;

%% load net
global sigmas; 
sigmas = 15/255.;  %noise level for FFDnet
load('FFDNet_gray.mat');
net = vl_simplenn_tidy(net);

%% load cameraman image
f = double(imread('cameraman.tif'))/255;
f = im2single(f);

%% set parameters for power method
it_num = 2000;    % number of iterations
freq = 1;      % output frequency

%% initalize power method
[M,N] = size(f);
u = f;

% save original mean and norm
orig_mean = mean(u(:));
u_removed = u - orig_mean;
orig_norm = norm(double(u_removed(:)));

% define containers
u_arr = zeros(M,N,it_num);
Rayleigh_arr = zeros(1,it_num);
theta_norm_arr = zeros(1,it_num);

%% run power method
figure;
imshow(u,[])
title('Initial condition');
drawnow;

tic;
for i = 1 : it_num
    res = vl_ffdnet_matlab(net,u);
    u_T = res(end).x;
         
    m_u = mean(u(:));
    
    % remove mean from u
    u = u - m_u;
    
    % remove mean from u_T
    u_T = u_T - mean(u_T(:));
      
    % compute inner products and norms
    u_u_T = sum(u(:).*u_T(:));
    norm_u = norm(u(:));
    norm_u_T = norm(u_T(:));
    
    % compute Rayleigh quotient
    Rayleigh_arr(i) = u_u_T / (norm_u^2);
    
    % compute angle in degrees
    theta_norm_arr(i)=acos(u_u_T / (norm_u * norm_u_T)) * 180/pi;

    % normalize by original norm   
    u_T = u_T / norm_u_T * orig_norm;
    
    % return to mean of u
    u_T = u_T + m_u;
      
    % update variables
    u = u_T;
    u_arr(:,:,i) = u;
    
    % output
    if mod(i, freq) == 0
        fprintf(['Finished ', num2str(i), ' iterations of the power method\n'])
        
        imshow(u,[]);
        title(['Iteration ', num2str(i)]);
        drawnow;        
    end
end
toc;
%% plot validation measures
figure;
plot(Rayleigh_arr,'k','LineWidth',3)
xlabel('\bf iterations','FontSize',40,'Interpreter','latex');
ax = gca;
ax.LineWidth = 1;
set(gca, 'FontSize', 30)
title('Rayleigh quotients');

figure;
plot(1:it_num, theta_norm_arr,'k','LineWidth',3)
xlabel('\bf iterations','FontSize',40,'Interpreter','latex');
ax = gca;
ax.LineWidth = 1;
set(gca, 'FontSize', 30)
title('Angles');
