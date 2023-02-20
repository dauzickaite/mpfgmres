function [BE,FE,iter,normZk,nZkMRxdiff,psiA,psiL,rho,zeta,x] = ...
    solveFGMRES(A, b, x0 ,tol, maxit, restart, U, L, P, u, uA, uR ,uL, n, xtrue, xtruen)

    Pright = @(x,convprc) Plu(x,convprc,eye(n),U,P,u,uR); 
    Pleft = @(x,convprc) Plu(x,convprc,L,eye(n),P,u,uL);

    Pright_ex = @(x,convprc) Plu(x,convprc,eye(n),U,P,u,'exact'); 
    Pleft_ex = @(x,convprc) Plu(x,convprc,L,eye(n),P,u,'exact');

    [x,~,~,iter,~, ~, Z, ~, ~, uApsiA, uLpsiL] = ...
        mpfgmres(A,b,x0,tol,maxit,restart,Pright,Pleft,u,uA,Pright_ex,Pleft_ex,mp(U));

    Afull = full(A);
    res = mp(b)-mp(Afull)*mp(x);
    nb = norm(mp(b));
    BEden = nb+norm(mp(Afull))*norm(mp(x));
    BE = norm(res)/BEden;
    normZk = norm(mp(Z));

    FE = norm(mp(x) - xtrue)/xtruen;

    Linv = inv(mp(L));
    MLELn = norm(abs(mp(L))*abs(Linv)*abs(mp(L))*abs(Linv));
    switch uL 
        case 'quad'
             zeta1 = 0.5*eps(u)+(2^(-226))*MLELn;
        case {'double','single'}
            zeta1 = 0.5*eps(u)+0.25*eps(uL)^2*MLELn;
        case 'half'
            zeta1 = 0.5*eps(u)+(2^(-22))*MLELn;
    end
      
    normU = norm(mp(U,64));
    MZK = normU*normZk;
    
    nbprec = norm(mp(L)\mp(b));
    nAprec = norm(mp(L)\mp(A));
    BEprec_den = nbprec+nAprec*norm(mp(x));
    nZkMRxdiff = normZk * norm(mp(U)*mp(x-x0));
    zeta = (zeta1*nbprec + (0.5*eps(u)+ max(uApsiA) + max(uLpsiL))*...
        nAprec*(norm(mp(x0)) + nZkMRxdiff))/BEprec_den;
    
    psiA = uApsiA./(0.5*eps(uA));
    switch uL 
        case 'quad'
            psiL = uLpsiL./(2^(-113));
        case {'double','single'}
            psiL = uLpsiL./(0.5*eps(uL));
        case 'half'
            psiL = uLpsiL./(2^(-11));
    end
    
    Uinv = inv(mp(U,64));
    condUnormUinv = norm(abs(Uinv)*abs(mp(U))*abs(Uinv))*normU;
    
    switch uR 
        case 'quad'
            rho = 0.5*eps(u)*MZK+(2^(-226))*condUnormUinv;
        case {'double','single'}
            rho = 0.5*eps(u)*MZK+0.25*eps(uR)^2*condUnormUinv;
        case 'half'
            rho = 0.5*eps(u)*MZK+(2^(-22))*condUnormUinv;
    end
    