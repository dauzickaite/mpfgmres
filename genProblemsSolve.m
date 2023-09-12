function genProblemsSolve(c,solveAll,dense, problem, solveLeft, solveRight)

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

solver = 'fgmres';
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

ER_MRinvapp= min(norm(abs(mp(U)\eye(n))*abs(mp(U))),norm(abs(mp(U))*abs(mp(U)\eye(n))));
fprintf('Approx. for ||E_R||/||M_R^(-1)|| %d \n',round(ER_MRinvapp))


%% solve, split-preconditioned
precond = 'split';

if solveAll
    if dense || strcmp(problem,'rajat14') 
        uL = 'half'; uR = 'half';
        [BE_hh,FE_hh,iter_hh,ZK_hh,ZkMRxdiff_hh, psiA_hh,psiL_hh,rho_hh, zeta_hh, x_hh] =...
            solveFGMRES(A, b, x0 ,tol, maxit, restart, U, L, P, u, uA, uR ,uL, n, xtrue, xtruen,precond,solver);

        uL = 'single'; uR = 'half';
        [BE_sh,FE_sh,iter_sh, ZK_sh, ZkMRxdiff_sh,psiA_sh,psiL_sh,rho_sh, zeta_sh, x_sh] = ...
            solveFGMRES(A, b, x0 ,tol, maxit, restart, U, L, P, u, uA, uR ,uL, n, xtrue, xtruen,precond,solver);
        
        uL = 'double'; uR = 'half';
        [BE_dh,FE_dh,iter_dh, ZK_dh, ZkMRxdiff_dh, psiA_dh,psiL_dh,rho_dh, zeta_dh, x_dh] =...
            solveFGMRES(A, b, x0 ,tol, maxit, restart, U, L, P, u, uA, uR ,uL, n, xtrue, xtruen,precond,solver);
        
        uL = 'quad'; uR = 'half';
        [BE_qh,FE_qh,iter_qh, ZK_qh,ZkMRxdiff_qh, psiA_qh,psiL_qh,rho_qh, zeta_qh, x_qh] =...
            solveFGMRES(A, b, x0 ,tol, maxit, restart, U, L, P, u, uA, uR ,uL, n, xtrue, xtruen,precond,solver);
    else
        BE_qh = NaN; BE_dh = NaN; BE_sh = NaN; BE_hh = NaN;
        FE_qh = NaN; FE_dh = NaN; FE_sh = NaN; FE_hh = NaN;
        iter_qh = NaN; iter_dh = NaN; iter_sh = NaN; iter_hh = NaN;
        rho_qh = NaN; rho_dh = NaN; rho_sh = NaN; rho_hh = NaN; 
        zeta_qh = NaN; zeta_dh= NaN; zeta_sh= NaN; zeta_hh= NaN;
    end
    
    uL = 'half'; uR = 'single';
    [BE_hs,FE_hs,iter_hs,ZK_hs,ZkMRxdiff_hs,psiA_hs,psiL_hs,rho_hs,zeta_hs, x_hs] =...
        solveFGMRES(A, b, x0 ,tol, maxit, restart, U, L, P, u, uA, uR ,uL, n, xtrue, xtruen,precond,solver);

    uL = 'half'; uR = 'double';
    [BE_hd,FE_hd,iter_hd,ZK_hd,ZkMRxdiff_hd,psiA_hd,psiL_hd,rho_hd,zeta_hd, x_hd] =...
        solveFGMRES(A, b, x0 ,tol, maxit, restart, U, L, P, u, uA, uR ,uL, n, xtrue, xtruen,precond,solver);

    uL = 'half'; uR = 'quad';
    [BE_hq,FE_hq,iter_hq,ZK_hq, ZkMRxdiff_hq, psiA_hq,psiL_hq,rho_hq,zeta_hq, x_hq] = ...
        solveFGMRES(A, b, x0 ,tol, maxit, restart, U, L, P, u, uA, uR ,uL, n, xtrue, xtruen,precond,solver);

    uL = 'single'; uR = 'single';
    [BE_ss,FE_ss,iter_ss,ZK_ss,ZkMRxdiff_ss,psiA_ss,psiL_ss,rho_ss,zeta_ss, x_ss] =...
        solveFGMRES(A, b, x0 ,tol, maxit, restart, U, L, P, u, uA, uR ,uL, n, xtrue, xtruen,precond,solver);

    uL = 'single'; uR = 'quad';
    [BE_sq,FE_sq,iter_sq,ZK_sq,ZkMRxdiff_sq,psiA_sq,psiL_sq,rho_sq,zeta_sq,x_sq] =...
        solveFGMRES(A, b, x0 ,tol, maxit, restart, U, L, P, u, uA, uR ,uL, n, xtrue, xtruen,precond,solver);

    uL = 'double'; uR = 'double';
    [BE_dd,FE_dd,iter_dd,ZK_dd,ZkMRxdiff_dd,psiA_dd,psiL_dd,rho_dd,zeta_dd, x_dd] =...
        solveFGMRES(A, b, x0 ,tol, maxit, restart, U, L, P, u, uA, uR ,uL, n, xtrue, xtruen,precond,solver);

    uL = 'double'; uR = 'quad';
    [BE_dq,FE_dq,iter_dq,ZK_dq,ZkMRxdiff_dq,psiA_dq,psiL_dq,rho_dq,zeta_dq,x_dq] =...
        solveFGMRES(A, b, x0 ,tol, maxit, restart, U, L, P, u, uA, uR ,uL, n, xtrue, xtruen,precond,solver);
    
    uL = 'quad'; uR = 'single';
    [BE_qs,FE_qs,iter_qs,ZK_qs,ZkMRxdiff_qs,psiA_qs,psiL_qs,rho_qs,zeta_qs,x_qs] = ...
        solveFGMRES(A, b, x0 ,tol, maxit, restart, U, L, P, u, uA, uR ,uL, n, xtrue, xtruen,precond,solver);

    uL = 'quad'; uR = 'double';
    [BE_qd,FE_qd,iter_qd,ZK_qd,ZkMRxdiff_qd,psiA_qd,psiL_qd,rho_qd,zeta_qd,x_qd] =...
        solveFGMRES(A, b, x0 ,tol, maxit, restart, U, L, P, u, uA, uR ,uL, n, xtrue, xtruen,precond,solver);

    uL = 'quad'; uR = 'quad';
    [BE_qq,FE_qq,iter_qq,ZK_qq,ZkMRxdiff_qq,psiA_qq,psiL_qq,rho_qq,zeta_qq,x_qq] =...
        solveFGMRES(A, b, x0 ,tol, maxit, restart, U, L, P, u, uA, uR ,uL, n, xtrue, xtruen,precond,solver);

end

uL = 'single'; uR = 'double';
[BE_sd,FE_sd,iter_sd,ZK_sd,ZkMRxdiff_sd,psiA_sd,psiL_sd,rho_sd,zeta_sd,x_sd] =...
    solveFGMRES(A, b, x0 ,tol, maxit, restart, U, L, P, u, uA, uR ,uL, n, xtrue, xtruen,precond,solver);

fprintf('u_L  %s, u_R  %s \n',uL,uR)
fprintf('IC %d, BE %.2e, FE %.2e, zeta %.2e, \n',iter_sd,BE_sd,FE_sd,zeta_sd)
fprintf('||Z_k||||M_R (x_k - x_0) || %.2e, psi_A %.2e,',ZkMRxdiff_sd(end),max(psiA_sd))
fprintf('psi_L %.2e, rho %.2e \n',max(psiL_sd),max(rho_sd))

uL = 'double'; uR = 'single';
[BE_ds,FE_ds,iter_ds,ZK_ds,ZkMRxdiff_ds,psiA_ds,psiL_ds,rho_ds,zeta_ds,x_ds] = ...
    solveFGMRES(A, b, x0 ,tol, maxit, restart, U, L, P, u, uA, uR ,uL, n, xtrue, xtruen,precond,solver);

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


%% solve left
if solveLeft
    precond = 'left';
    uR = 'double';
    
    normLUA = norm(mp(U)\(mp(L)\mp(P*A)));

    condL = norm(abs(mp(L)\eye(n))*abs(mp(L)));
    condU = norm(abs(mp(U)\eye(n))*abs(mp(U)));

    fprintf('c = %d \n', c)
    fprintf('left-preconditioning \n')
    fprintf('kappa(U^(-1)L^(-1)A) %.2e \n',cond(mp(U)\(mp(L)\mp(P*A))))
    fprintf('psi_A bound %.2e \n', norm(abs(mp(U)\(mp(L)\P))*abs(A))/normLUA )
    fprintf('psi_L bound %.2e \n',...
        norm(mp(U)\eye(n))*norm(mp(L)\mp(P*A))*condL/normLUA + condU)

    
    if solveAll
        
        uL = 'half'; 
        [BE_h_l,FE_h_l,iter_h_l,ZK_h_l,ZkMRxdiff_h_l,psiA_h_l,psiL_h_l,rho_h_l,zeta_h_l,x_h_l,Z_h_l] = ...
            solveFGMRES(A, b, x0 ,tol, maxit, restart, U, L, P, u, uA, uR ,uL, n, xtrue, xtruen,precond,solver);

        uL = 'single'; 
        [BE_s_l,FE_s_l,iter_s_l,ZK_s_l,ZkMRxdiff_s_l,psiA_s_l,psiL_s_l,rho_s_l,zeta_s_l,x_s_l,Z_s_l] = ...
            solveFGMRES(A, b, x0 ,tol, maxit, restart, U, L, P, u, uA, uR ,uL, n, xtrue, xtruen,precond,solver);

        uL = 'double'; 
        [BE_d_l,FE_d_l,iter_d_l,ZK_d_l,ZkMRxdiff_d_l,psiA_d_l,psiL_d_l,rho_d_l,zeta_d_l,x_d_l,Z_d_l] = ...
            solveFGMRES(A, b, x0 ,tol, maxit, restart, U, L, P, u, uA, uR ,uL, n, xtrue, xtruen,precond,solver);

        uL = 'quad'; 
        [BE_q_l,FE_q_l,iter_q_l,ZK_q_l,ZkMRxdiff_q_l,psiA_q_l,psiL_q_l,rho_q_l,zeta_q_l,x_q_l,Z_q_l] = ...
            solveFGMRES(A, b, x0 ,tol, maxit, restart, U, L, P, u, uA, uR ,uL, n, xtrue, xtruen,precond,solver);



        % heatmaps
        xvalues = {'Half','Single','Double', 'Quad'};

        % backward error
        BEtbl_left = [BE_h_l, BE_s_l, BE_d_l, BE_q_l];

        % forward error
        FEtbl_left = [FE_h_l, FE_s_l, FE_d_l, FE_q_l];

        % iteration count
        itertbl_left = [iter_h_l, iter_s_l, iter_d_l, iter_q_l];

        % 3-in-1 plot
        figure;
        tiledlayout(3,1)

        nexttile
        hBE_left = heatmap(xvalues,{'BE'},BEtbl_left);
        hBE_left.CellLabelFormat = '%.0e';
        set(gca,'ColorScaling','log')
        set(gca, 'FontSize',50)


        nexttile
        hFE_left = heatmap(xvalues,{'FE'},FEtbl_left);
        hFE_left.CellLabelFormat = '%.0e';
        set(gca,'ColorScaling','log')
        set(gca, 'FontSize',50)


        nexttile
        heatmap(xvalues,{'IC'},itertbl_left);
        set(gca,'ColorScaling','log')
        set(gca, 'FontSize',50)
        
    else
        uL = 'single'; 
        [BE_s_l,FE_s_l,iter_s_l,ZK_s_l,ZkMRxdiff_s_l,psiA_s_l,psiL_s_l,rho_s_l,zeta_s_l,x_s_l,Z_s_l] = ...
            solveFGMRES(A, b, x0 ,tol, maxit, restart, U, L, P, u, uA, uR ,uL, n, xtrue, xtruen,precond,solver);

        uL = 'double'; 
        [BE_d_l,FE_d_l,iter_d_l,ZK_d_l,ZkMRxdiff_d_l,psiA_d_l,psiL_d_l,rho_d_l,zeta_d_l,x_d_l,Z_d_l] = ...
            solveFGMRES(A, b, x0 ,tol, maxit, restart, U, L, P, u, uA, uR ,uL, n, xtrue, xtruen,precond,solver);

    end
    fprintf('c = %d \n', c)
    fprintf('left-preconditioning \n')
    fprintf('u_L = single \n')
    fprintf('IC %d \n', iter_s_l)
    fprintf('BE %.2e \n',  BE_s_l)
    fprintf('FE %.2e \n',  FE_s_l)
    fprintf('zeta %.2e \n',  zeta_s_l)
    fprintf('psi_A %.2e \n',  max(psiA_s_l))
    fprintf('psi_L %.2e \n',  max(psiL_s_l))


    fprintf('u_L = double \n')
    fprintf('IC %d \n', iter_d_l)
    fprintf('BE %.2e \n',  BE_d_l)
    fprintf('FE %.2e \n',  FE_d_l)
    fprintf('zeta %.2e \n',  zeta_d_l)
    fprintf('psi_A %.2e \n',  max(psiA_d_l))
    fprintf('psi_L %.2e \n',  max(psiL_d_l))

end
%% solve right
if solveRight
    precond = 'right';
    uL = 'double';
    
    fprintf('right-preconditioning \n')
    fprintf('kappa(AU^(-1)L^(-1)) %.2e \n',cond((mp(A)/mp(U)/mp(L))*P))
    fprintf('Approx. for || E_R || / || M_R^(-1) || %.2e \n',...
       norm(abs(mp(P'*L))*abs(mp(L)\P))+ condU)

     if solveAll

        uR = 'half'; 
        [BE_h_r,FE_h_r,iter_h_r,ZK_h_r,ZkMRxdiff_h_r,psiA_h_r,psiL_h_r,rho_h_r,zeta_h_r,x_h_r,Z_h_r] = ...
            solveFGMRES(A, b, x0 ,tol, maxit, restart, U, L, P, u, uA, uR ,uL, n, xtrue, xtruen,precond,solver);

        uR = 'single'; 
        [BE_s_r,FE_s_r,iter_s_r,ZK_s_r,ZkMRxdiff_s_r,psiA_s_r,psiL_s_r,rho_s_r,zeta_s_r,x_s_r,Z_s_r] = ...
            solveFGMRES(A, b, x0 ,tol, maxit, restart, U, L, P, u, uA, uR ,uL, n, xtrue, xtruen,precond,solver);

        uR = 'double'; 
        [BE_d_r,FE_d_r,iter_d_r,ZK_d_r,ZkMRxdiff_d_r,psiA_d_r,psiL_d_r,rho_d_r,zeta_d_r,x_d_r,Z_d_r] = ...
            solveFGMRES(A, b, x0 ,tol, maxit, restart, U, L, P, u, uA, uR ,uL, n, xtrue, xtruen,precond,solver);

        uR = 'quad'; 
        [BE_q_r,FE_q_r,iter_q_r,ZK_q_r,ZkMRxdiff_q_r,psiA_q_r,psiL_q_r,rho_q_r,zeta_q_r,x_q_r,Z_q_r] = ...
            solveFGMRES(A, b, x0 ,tol, maxit, restart, U, L, P, u, uA, uR ,uL, n, xtrue, xtruen,precond,solver);

        % heatmaps
        xvalues = {'Half','Single','Double', 'Quad'};

        % 3-in-1 plot
        figure;
        tiledlayout(3,1)

        nexttile
        BEtbl_right = [BE_h_r, BE_s_r, BE_d_r, BE_q_r];
        hBE_right = heatmap(xvalues,{'BE'},BEtbl_right);
        hBE_right.CellLabelFormat = '%.0e';
        set(gca,'ColorScaling','log')
        set(gca, 'FontSize',50)


        nexttile
        FEtbl_right = [FE_h_r, FE_s_r, FE_d_r, FE_q_r];
        hFE_right = heatmap(xvalues,{'FE'},FEtbl_right);
        hFE_right.CellLabelFormat = '%.0e';
        set(gca,'ColorScaling','log')
        set(gca, 'FontSize',50)


        nexttile
        itertbl_right = [iter_h_r, iter_s_r, iter_d_r, iter_q_r];
        heatmap(xvalues,{'IC'},itertbl_right);

        set(gca,'ColorScaling','log')
        set(gca, 'FontSize',50)
        
     else
         
         uR = 'single'; 
        [BE_s_r,FE_s_r,iter_s_r,ZK_s_r,ZkMRxdiff_s_r,psiA_s_r,psiL_s_r,rho_s_r,zeta_s_r,x_s_r,Z_s_r] = ...
            solveFGMRES(A, b, x0 ,tol, maxit, restart, U, L, P, u, uA, uR ,uL, n, xtrue, xtruen,precond,solver);

        uR = 'double'; 
        [BE_d_r,FE_d_r,iter_d_r,ZK_d_r,ZkMRxdiff_d_r,psiA_d_r,psiL_d_r,rho_d_r,zeta_d_r,x_d_r,Z_d_r] = ...
            solveFGMRES(A, b, x0 ,tol, maxit, restart, U, L, P, u, uA, uR ,uL, n, xtrue, xtruen,precond,solver);
     end
     
    fprintf('c = %d \n', c)
    fprintf('right-preconditioning \n')

    fprintf('u_R = single \n')
    fprintf('IC %d \n', iter_s_r)
    fprintf('BE %.2e \n',  BE_s_r)
    fprintf('FE %.2e \n',  FE_s_r)
    fprintf('zeta %.2e \n',  zeta_s_r)
    fprintf('||Z_k||||M_R (x_k - x_0) || %.2e \n',ZkMRxdiff_s_r(end))
    fprintf('psi_A %.2e \n',  max(psiA_s_r))
    fprintf('rho %.2e \n', rho_s_r )

    fprintf('u_R = double \n')
    fprintf('IC %d \n', iter_d_r)
    fprintf('BE %.2e \n',  BE_d_r)
    fprintf('FE %.2e \n',  FE_d_r)
    fprintf('zeta %.2e \n',  zeta_d_r)
    fprintf('||Z_k||||M_R (x_k - x_0) || %.2e \n',ZkMRxdiff_d_r(end))
    fprintf('psi_A %.2e \n',  max(psiA_d_r))
    fprintf('rho %.2e \n', rho_d_r )

     
end

