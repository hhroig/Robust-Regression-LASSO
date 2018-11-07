%% MAIN script: Runs all exposed variants of Robust Linear Regression and relationship with LASSO
%  It generates a simulated dataset to check the behaviour of estimations
%  under uncertainty of the observed data (Robust Linear Regression).
%  It is assumed that the disturbances are feature-wise and are bounded.
%  We propose 3 approaches: (1) No knowledge about the distributions of the
%  delta_i's or the boundaries c_i of their 2-norms; (2) Knowledge of c_i or
%  maximum boundary for all the c_i; and (3) Knowledge of the distribution
%  of every delta_i.
% %  
% %                     
% %   Optimization Final Project (November, 2018)
% %   Harold A. Hernández Roig (hahernan@est-econ.uc3m.es)

% Simulated Data:
n = 200;
p = 200;
[X,Y, weights] = sample_data(n,p,10);
 

% Perturbation matrix "Delta":
rng default % For reproducibility

% Disturbance is Normaly Distr.
mu = 0;
sigma2 = 0.1;
D = sigma2.*randn(size(X)) + mu; % columns are ~N(mu,sigma^2) vectors!

% Disturbance is Uniformly Distr.
% a = 0;
% b = 0.1;
% D = a + (b-a).*rand(size(X)); % columns are ~U(0,0.1) vectors!

% Perturbed design matrix!
XX = X + D;

% True values for c_i
c = vecnorm(D);

%% Initialize Parameter (perhaps not needed when using MATLAB's built-in)
% choose the maximum lambda such that betas = 0 is the only optimal sol.
% lambda_max = mx_j | 1/n <x_j, y > |% lambda_max = max(abs(1/n*Y'*X));
% 
% % G is the grid of lambdas:
% quant_lambdas = 100;
% G = linspace(lambda_max, 0, quant_lambdas);


%% Get LASSO Solutions

% When no information about the c_i nor the distributions of the delta_i's
[betas2,FitInfo2] = lasso(XX,Y,'CV',10);

lambda_minMSPE = FitInfo2.LambdaMinMSE; % optimum lambda for min MSPE
lambda_minMSPESE = FitInfo2.Lambda1SE; % optimum lambda for min MSPE + 1SE

% When assuming c = max(c_i) is a permisible bound
[betas3,FitInfo3] = lasso(XX,Y,'Lambda', max(c));

% Checking MSPE vs. value of Lambda
[ax, fig]=lassoPlot(betas2,FitInfo2,'PlotType','CV');
%legend('show') % Show legend
hold on
line([max(c) max(c)],get(ax,'YLim'),'Color',[1 0 0], 'LineStyle', '--')
legend({'MSPE with Error Bars', '$\lambda$ for $\min$ MSPE', '$\lambda$ for $\min$ MSPE+1S.E', '$\max c_i$'},'Interpreter','latex')


%% Solution when the values c_i are known!
% We do know them since they would be the 2-norm of every column in D.

fun = @(beta) norm(Y - X*beta) + c*abs(beta);
beta0 = zeros(p,1);

options = optimset('MaxFunEvals',100^100, 'MaxIter', 100^100);
sol_beta = fminsearch(fun,beta0,options);

%% Monte Carlo Sampling & Bisection for admissible disturbances
% Case of known distributions for delta_i's but no knowledge about c_i.
% We assume Normal Distr. (drops for Uniform Distr. are commented) 
S = 1000;
rng('shuffle') 

alpha = 0.05; % 1-alpha is confidence in satisfying uncertainty set
epsilon = 10^(-8); % tolerance for bisection

est_c = zeros(1,p); % these are the c_i^* of the report

for i = 1:p
    delta_i_matrix = zeros(p, S);
    for s = 1:S
        delta_i_matrix(:,s) = sigma2.*randn(p,1) + mu; % columns are ~N(mu,sigma^2) vectors!        
        %delta_i_matrix(:,s) = a + (b-a).*rand(p,1); % columns are ~U(a,b) vectors!  
    end
    deltas_norms = vecnorm(delta_i_matrix); % for every generated vector: || delta_i^s ||_2
    
    % starts bisection!
    c_l = 0;
    c_u = max(deltas_norms);
    
    f = @(x) sum(deltas_norms > x)/S - alpha;
    fu = f(c_u);
    fl = f(c_l);    
    cc = (c_l + c_u)/2;
    
    while abs(f(cc)) > epsilon
        cc = (c_l + c_u)/2;
        if f(c_l)*f(cc) > 0
            c_l = cc;
        else
            c_u = cc;
        end

    end
    est_c(i) = cc;    
end

% Now it is possible to directly compute the solution by saying c_i = c_i^*:

fun_est = @(beta) norm(Y - X*beta) + est_c*abs(beta);
beta0 = zeros(p,1);

options = optimset('MaxFunEvals',100^100, 'MaxIter', 100^100);
sol_beta2 = fminsearch(fun_est,beta0,options);

% Another option (not expensive) is to assume  c_i = cte = max(c_i^*) and use LASSO:
[betas4,FitInfo4] = lasso(XX,Y,'Lambda', max(est_c));