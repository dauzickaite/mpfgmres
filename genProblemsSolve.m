function genProblemsSolve(c,solveAll,dense, problem)

% Generates and solves Ax=b via mixed precision FGMRES.
% Split preconditioners are used.

mp.Digits(64);

if dense    
    fprintf('c = %d \n',c)

    n = 2*1e2;
    gamma = 1;
    nm1 = n-1;
    d = 10.^(-c*((0:nm1)./nm1).^gamma); % || A ||_2 = 1

    rng(123)
    [U,~] = qr(rand(n));
    [V,~] =  qr(rand(n));
    A = U*diag(d)*V;
    
else
    fprintf('problem: %s \n',problem)
    load([problem,'.mat'])
    A = Problem.A;
    n = size(A,1);
end

Afull = full(A);     
rng(456) 
b = rand(n,1);

xtrue = mp(A)\mp(b);
xtruen = norm(xtrue);

u = 'double'; % tolerance set to 2*eps(u) 
uA = 'double';

x0 = zeros(n,1);

tol = 2*eps(u);
maxit = 1; % max number of restarts+1
restart = n; % number of iterations before restarting
% total (max) number of FGMRES iterations = maxit * restart

    % unpreconditioned solve
%     [BE_unp,FE_unp,iter_unp,zeta1_unp,zeta2_unp,ZK_unp,ZkMRxdiff_unp,...
%         psiA_unp,psiL_unp,rho_unp, zeta_new_unp, x_unp] =...
%                 solveFGMRES(A, b, x0 ,tol, maxit, restart, eye(n), eye(n),...
%                 eye(n), u, u, u ,u, n, xtrue, xtruen,1);
% 
     
%% generate preconditioner
if dense
    if c < 6
        [L,U,P] = lu(mp(A,4));
    else
        [L,U,P] = lu(single(A));
    end
else
    [L,U,P] = lu(single(Afull));
end

kappaA = cond(mp(Afull));
kappaL = cond(mp(L));
kappaLinvA = cond(mp(L)\(P*mp(Afull)));
kappaLinvAUinv = cond((mp(L)\(P*mp(Afull)))/mp(U));
kappaU = cond(mp(U));
minsU = min(svd(U));

nLA = norm(mp(L)\(P*mp(Afull)));
psiAratioapprox = norm(abs(mp(L))\abs(P*mp(Afull)))/nLA;

fprintf('kappa(A) %.2e, kappa(M_L^(-1)*A) %.2e, ',kappaA,kappaLinvA)
fprintf('kappa(M_L^(-1)*A*M_R^(-1)) %.2e, kappa(M_R) %.2e, \n',kappaLinvAUinv,kappaU)
fprintf('kappa(M_L) and psi_L bound %.2e, psi_A bound %.2e \n',kappaL,psiAratioapprox)


%% solve
if solveAll
    if minsU > 6e-5
        uL = 'half'; uR = 'half';
        [BE_hh,FE_hh,iter_hh,ZK_hh,ZkMRxdiff_hh, psiA_hh,psiL_hh,rho_hh, zeta_hh, x_hh] =...
            solveFGMRES(A, b, x0 ,tol, maxit, restart, U, L, P, u, uA, uR ,uL, n, xtrue, xtruen);

        uL = 'single'; uR = 'half';
        [BE_sh,FE_sh,iter_sh, ZK_sh, ZkMRxdiff_sh,psiA_sh,psiL_sh,rho_sh, zeta_sh, x_sh] = ...
            solveFGMRES(A, b, x0 ,tol, maxit, restart, U, L, P, u, uA, uR ,uL, n, xtrue, xtruen);
        
        uL = 'double'; uR = 'half';
        [BE_dh,FE_dh,iter_dh, ZK_dh, ZkMRxdiff_dh, psiA_dh,psiL_dh,rho_dh, zeta_dh, x_dh] =...
            solveFGMRES(A, b, x0 ,tol, maxit, restart, U, L, P, u, uA, uR ,uL, n, xtrue, xtruen);
        
        uL = 'quad'; uR = 'half';
        [BE_qh,FE_qh,iter_qh, ZK_qh,ZkMRxdiff_qh, psiA_qh,psiL_qh,rho_qh, zeta_qh, x_qh] =...
            solveFGMRES(A, b, x0 ,tol, maxit, restart, U, L, P, u, uA, uR ,uL, n, xtrue, xtruen);
    else
        BE_qh = NaN; BE_dh = NaN; BE_sh = NaN; BE_hh = NaN;
        FE_qh = NaN; FE_dh = NaN; FE_sh = NaN; FE_hh = NaN;
        iter_qh = NaN; iter_dh = NaN; iter_sh = NaN; iter_hh = NaN;
        rho_qh = NaN; rho_dh = NaN; rho_sh = NaN; rho_hh = NaN; 
        zeta_qh = NaN; zeta_dh= NaN; zeta_sh= NaN; zeta_hh= NaN;
    end
    
    uL = 'half'; uR = 'single';
    [BE_hs,FE_hs,iter_hs,ZK_hs,ZkMRxdiff_hs,psiA_hs,psiL_hs,rho_hs,zeta_hs, x_hs] =...
        solveFGMRES(A, b, x0 ,tol, maxit, restart, U, L, P, u, uA, uR ,uL, n, xtrue, xtruen);

    uL = 'half'; uR = 'double';
    [BE_hd,FE_hd,iter_hd,ZK_hd,ZkMRxdiff_hd,psiA_hd,psiL_hd,rho_hd,zeta_hd, x_hd] =...
        solveFGMRES(A, b, x0 ,tol, maxit, restart, U, L, P, u, uA, uR ,uL, n, xtrue, xtruen);

    uL = 'half'; uR = 'quad';
    [BE_hq,FE_hq,iter_hq,ZK_hq, ZkMRxdiff_hq, psiA_hq,psiL_hq,rho_hq,zeta_hq, x_hq] = ...
        solveFGMRES(A, b, x0 ,tol, maxit, restart, U, L, P, u, uA, uR ,uL, n, xtrue, xtruen);

    uL = 'single'; uR = 'single';
    [BE_ss,FE_ss,iter_ss,ZK_ss,ZkMRxdiff_ss,psiA_ss,psiL_ss,rho_ss,zeta_ss, x_ss] =...
        solveFGMRES(A, b, x0 ,tol, maxit, restart, U, L, P, u, uA, uR ,uL, n, xtrue, xtruen);

    uL = 'single'; uR = 'quad';
    [BE_sq,FE_sq,iter_sq,ZK_sq,ZkMRxdiff_sq,psiA_sq,psiL_sq,rho_sq,zeta_sq,x_sq] =...
        solveFGMRES(A, b, x0 ,tol, maxit, restart, U, L, P, u, uA, uR ,uL, n, xtrue, xtruen);

    uL = 'double'; uR = 'double';
    [BE_dd,FE_dd,iter_dd,ZK_dd,ZkMRxdiff_dd,psiA_dd,psiL_dd,rho_dd,zeta_dd, x_dd] =...
        solveFGMRES(A, b, x0 ,tol, maxit, restart, U, L, P, u, uA, uR ,uL, n, xtrue, xtruen);

    uL = 'double'; uR = 'quad';
    [BE_dq,FE_dq,iter_dq,ZK_dq,ZkMRxdiff_dq,psiA_dq,psiL_dq,rho_dq,zeta_dq,x_dq] =...
        solveFGMRES(A, b, x0 ,tol, maxit, restart, U, L, P, u, uA, uR ,uL, n, xtrue, xtruen);
    
    uL = 'quad'; uR = 'single';
    [BE_qs,FE_qs,iter_qs,ZK_qs,ZkMRxdiff_qs,psiA_qs,psiL_qs,rho_qs,zeta_qs,x_qs] = ...
        solveFGMRES(A, b, x0 ,tol, maxit, restart, U, L, P, u, uA, uR ,uL, n, xtrue, xtruen);

    uL = 'quad'; uR = 'double';
    [BE_qd,FE_qd,iter_qd,ZK_qd,ZkMRxdiff_qd,psiA_qd,psiL_qd,rho_qd,zeta_qd,x_qd] =...
        solveFGMRES(A, b, x0 ,tol, maxit, restart, U, L, P, u, uA, uR ,uL, n, xtrue, xtruen);

    uL = 'quad'; uR = 'quad';
    [BE_qq,FE_qq,iter_qq,ZK_qq,ZkMRxdiff_qq,psiA_qq,psiL_qq,rho_qq,zeta_qq,x_qq] =...
        solveFGMRES(A, b, x0 ,tol, maxit, restart, U, L, P, u, uA, uR ,uL, n, xtrue, xtruen);

end

uL = 'single'; uR = 'double';
[BE_sd,FE_sd,iter_sd,ZK_sd,ZkMRxdiff_sd,psiA_sd,psiL_sd,rho_sd,zeta_sd,x_sd] =...
    solveFGMRES(A, b, x0 ,tol, maxit, restart, U, L, P, u, uA, uR ,uL, n, xtrue, xtruen);

fprintf('u_L  %s, u_R  %s \n',uL,uR)
fprintf('IC %d, BE %.2e, FE %.2e, zeta %.2e, \n',iter_sd,BE_sd,FE_sd,zeta_sd)
fprintf('||Z_k||||M_R (x_k - x_0) || %.2e, psi_A %.2e,',ZkMRxdiff_sd(end),max(psiA_sd))
fprintf('psi_L %.2e, rho %.2e \n',max(psiL_sd),max(rho_sd))

uL = 'double'; uR = 'single';
[BE_ds,FE_ds,iter_ds,ZK_ds,ZkMRxdiff_ds,psiA_ds,psiL_ds,rho_ds,zeta_ds,x_ds] = ...
    solveFGMRES(A, b, x0 ,tol, maxit, restart, U, L, P, u, uA, uR ,uL, n, xtrue, xtruen);

fprintf('u_L  %s, u_R  %s \n',uL,uR)
fprintf('IC %d, BE %.2e, FE %.2e, zeta %.2e, \n',iter_ds,BE_ds,FE_ds,zeta_ds)
fprintf('||Z_k||||M_R (x_k - x_0) || %.2e, psi_A %.2e,',ZkMRxdiff_ds(end),max(psiA_ds))
fprintf('psi_L %.2e, rho %.2e \n',max(psiL_ds),max(rho_ds))
    
%% heatmaps
if solveAll
    xvalues = {'Half','Single','Double', 'Quad'};
    yvalues = {'Quad','Double','Single','Half'};

    % backward error
    BEtbl = [BE_qh, BE_qs, BE_qd, BE_qq; BE_dh, BE_ds, BE_dd, BE_dq; ...
        BE_sh, BE_ss, BE_sd, BE_sq; BE_hh, BE_hs, BE_hd, BE_hq];
    figure; hBE = heatmap(xvalues,yvalues,BEtbl);
    hBE.Title = 'BE';
    hBE.XLabel = 'u_R';
    hBE.YLabel = 'u_L';
    hBE.CellLabelFormat = '%.0e';
    set(gca,'ColorScaling','log')
    set(gca, 'FontSize',50)

    % forward error 
    FEtbl = [FE_qh, FE_qs, FE_qd, FE_qq; FE_dh, FE_ds, FE_dd, FE_dq; ...
        FE_sh, FE_ss, FE_sd, FE_sq; FE_hh, FE_hs, FE_hd, FE_hq];
    figure; hFE = heatmap(xvalues,yvalues,FEtbl);
    hFE.Title = 'FE';
    hFE.XLabel = 'u_R';
    hFE.YLabel = 'u_L';
    hFE.CellLabelFormat = '%.0e';
    set(gca,'ColorScaling','log')
    set(gca, 'FontSize',50)

    % iteration count
    itertbl = [iter_qh, iter_qs, iter_qd, iter_qq; iter_dh, iter_ds,...
        iter_dd, iter_dq; iter_sh, iter_ss, iter_sd, iter_sq; iter_hh,...
        iter_hs, iter_hd, iter_hq];
    figure; hiter = heatmap(xvalues,yvalues,itertbl);
    hiter.Title = 'IC';
    hiter.XLabel = 'u_R';
    hiter.YLabel = 'u_L';
    set(gca, 'FontSize',50)

    if dense

        rhotbl = [ rho_hh, rho_hs, rho_hd, rho_hq];
        figure; hrho = heatmap(xvalues,{'All'},rhotbl);
        hrho.Title = '\rho';
        hrho.XLabel = 'u_R';
        hrho.YLabel = 'u_L';
        hrho.CellLabelFormat = '%.0e';
        hrho.Position = [0.15 0.25 0.7 0.2];
        set(gca,'ColorScaling','log')
        set(gca, 'FontSize',50)


        zeta_tbl = [zeta_qh, zeta_qs, zeta_qd, zeta_qq; zeta_dh, zeta_ds, zeta_dd, zeta_dq; ...
            zeta_sh, zeta_ss, zeta_sd, zeta_sq; zeta_hh, zeta_hs, zeta_hd, zeta_hq];
        figure; hzeta = heatmap(xvalues,yvalues,zeta_tbl);
        hzeta.Title = 'BE bound';
        hzeta.XLabel = 'u_R';
        hzeta.YLabel = 'u_L';
        hzeta.CellLabelFormat = '%.0e';
        set(gca,'ColorScaling','log')
        set(gca, 'FontSize',50)
    end
end    

