function x = CGsolver_LRDS(x0, param)
%
% reconstruction of dynamic graph signal reconstruction.
% minimization using non linear conjugate gradient iterations
% 
% Given the sampling model y = J*x, and the sparsifying transform T, 
% the pogram finds the x that minimizes the following objective function:
%
% f(x) = 0.5* ||J*x-y||^2 + 0.5* alpha * (Tx)'*L*(Tx) + 0.5* belta * ||x-z+w/belta||^2 
%
% Qiu Kai, 20151211
%

%fprintf('\n Non-linear conjugate gradient algorithm')
%fprintf('\n ---------------------------------------------\n')

% starting point
x=x0;

gradToll = 1e-4 ;
param.l1Smooth = 1e-15;	
k = 0;

g0 = grad(x,param);
dx = -g0;

while(1)
    % search for the optimal step size
    tmp = grad(dx,param)+param.y+param.belta*param.z-param.w;
    t=-(g0(:)'*dx(:))/(tmp(:)'*dx(:));

    % update x
	x = (x + t*dx);

	% print some numbers	
    if param.display,
        %fprintf(' ite = %d, cost = %f \n',k,f1);
    end
    
    %conjugate gradient calculation
	g1 = grad(x,param);
	bk = g1(:)'*g1(:)/(g0(:)'*g0(:)+eps);
	g0 = g1;
	dx =  - g1 + bk* dx;
	k = k + 1;
	
	% stopping criteria (to be improved)
	if (k > param.niter) || (norm(dx(:)) < gradToll)  %2范数，norm表示范数的求解
%         fprintf(' ite = %d\n',k);
        break;
    end
end
return;

function g = grad(x,param)%***********************************************

% part 1
Grad1 = param.J.*x-param.y;

% part 2
Grad2 = 0;
if param.alpha
    Grad2 =  param.T' * (param.L * (param.T * x));
end

% part 3
Grad3 = 0;
if param.belta
    Grad3 =  x - param.z + param.w/param.belta;
end

% composite gradient
g = Grad1 + param.alpha * Grad2 + param.belta * Grad3;

