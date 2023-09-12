function [BE,FE,iter,normZk,nZkMRxdiff,psiA,psiL,rho,zeta,x,Z] = ...
    solveFGMRES(A, b, x0 ,tol, maxit, restart, U, L, P, u, uA, uR ,uL, n,...
    xtrue, xtruen, precond, solver)
mp.Digits(64);

    switch precond
        case 'left'        
            % full left prec
            Pright = @(x,convprc) Plu(x,convprc,eye(n),eye(n),P,u,uR); 
            Pleft = @(x,convprc) Plu(x,convprc,L,U,P,u,uL);
            
            Pright_ex = @(x,convprc) Plu(x,convprc,eye(n),eye(n),P,u,'exact'); 
            Pleft_ex = @(x,convprc) Plu(x,convprc,L,U,P,u,'exact');
        case 'right'            
            % full right prec
            Pright = @(x,convprc) Plu(x,convprc,L,U,P,u,uR); 
            Pleft = @(x,convprc) Plu(x,convprc,eye(n),eye(n),P,u,uL);
            
            Pright_ex = @(x,convprc) Plu(x,convprc,L,U,P,u,'exact'); 
            Pleft_ex = @(x,convprc) Plu(x,convprc,eye(n),eye(n),P,u,'exact');
        case 'split'
            % split prec
            Pright = @(x,convprc) Plu(x,convprc,eye(n),U,P,u,uR); 
            Pleft = @(x,convprc) Plu(x,convprc,L,eye(n),P,u,uL);

            Pright_ex = @(x,convprc) Plu(x,convprc,eye(n),U,P,u,'exact'); 
            Pleft_ex = @(x,convprc) Plu(x,convprc,L,eye(n),P,u,'exact');
    end

    
    switch solver
        case 'fgmres'
            [x,resnorm,relres,iter,flag, ~, Z, y, ~, uApsiA, uLpsiL,~,~,~,~,BEfg] = ...
        mpfgmres(A,b,x0,tol,maxit,restart,Pright,Pleft,u,uA,Pright_ex,Pleft_ex,mp(U));
        case 'gmres'
            [x,resnorm,relres,iter,flag, ~, Z, y, ~, uApsiA, uLpsiL,~,~,~,~,BEfg] = ...
        mpgmres(A,b,x0,tol,maxit,restart,Pright,Pleft,u,uA,Pright_ex,Pleft_ex,mp(U));
    end

    
         
    Afull = full(A);
    res = mp(b)-mp(Afull)*mp(x);
    nb = norm(mp(b));
    nx = norm(mp(x));
    BEden = nb+norm(mp(Afull))*nx;
    BE = norm(res)/BEden;
    normZk = norm(mp(Z));

    FE = norm(mp(x) - xtrue)/xtruen;
    
    psiA = uApsiA./(0.5*eps(uA));
    switch uL 
        case 'quad'
            psiL = uLpsiL./(2^(-113));
        case {'double','single'}
            psiL = uLpsiL./(0.5*eps(uL));
        case 'half'
            psiL = uLpsiL./(2^(-11));
    end

%% computing the bound

    switch precond
        case 'left'        
            % full left prec
            rho =  0.5*eps(u);
            
            nbprec = norm(mp(U)\(mp(L)\mp(b)));
            nAprec = norm(mp(U)\(mp(L)\mp(A)));
            BEprec_den = nbprec+nAprec*norm(mp(x));

           
            MLELn = norm(abs((eye(n)/mp(U))* (eye(n)/mp(L)))*abs(L)*abs(eye(n)/mp(L))*abs(L*U)...
                +abs(eye(n)/mp(U))*abs(U)*abs((eye(n)/mp(U))* (eye(n)/mp(L)))*abs(L*U) );
            switch uL 
                case 'quad'
                     zeta1 = 0.5*eps(u)+(2^(-226))*MLELn;
                case {'double','single'}
                    zeta1 = 0.5*eps(u)+0.25*eps(uL)^2*MLELn;
                case 'half'
                    zeta1 = 0.5*eps(u)+(2^(-22))*MLELn;
            end

            
            zeta2 = (0.5*eps(u)+ max(uApsiA) + max(uLpsiL))*...
                    nAprec*(norm(mp(x0)) + norm(mp(x-x0)));
                         
            zeta = (zeta1*nbprec + zeta2)/BEprec_den;
            
                                    
             nZkMRxdiff = [];
            
        case 'right'            
            % full right prec
            nZkMRxdiff = normZk * norm(mp(L)*mp(U)*mp(x-x0));
            
            nA = norm(mp(Afull));
            zeta1 = 0.5*eps(u)*nb;
            zeta2 = (0.5*eps(u)+ max(uApsiA))*nA*(norm(mp(x0))+nZkMRxdiff);
            
            zeta = (zeta1 + zeta2)/BEden;
            
             
            normLU = norm(mp(L)*mp(U));
            MZK = normLU*normZk;
            condLcondUkappaLU = ...
                (norm(abs(eye(n)/mp(U))*abs(mp(U)))+norm(abs(mp(L))*abs(eye(n)/mp(L))))*cond(mp(U)*mp(L));
            switch uR 
                case 'quad'
                    rho = 0.5*eps(u)*MZK+(2^(-226))*condLcondUkappaLU;
                case {'double','single'}
                    rho = 0.5*eps(u)*MZK+0.25*eps(uR)^2*condLcondUkappaLU;
                case 'half'
                    rho = 0.5*eps(u)*MZK+(2^(-22))*condLcondUkappaLU;
                case 'mp2'
                    rho = [];
            end            

            
        case 'split'
            % split prec
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
                zeta2=(0.5*eps(u)+ max(uApsiA) + max(uLpsiL))*nAprec...
                    *(norm(mp(x0)) + nZkMRxdiff);
                zeta = (zeta1*nbprec + zeta2 ) /BEprec_den;

               
                condUnormUinv = norm(abs(eye(n)/mp(U))*abs(mp(U))*abs(eye(n)/mp(U)))*normU;

                switch uR 
                    case 'quad'
                        rho = 0.5*eps(u)*MZK+(2^(-226))*condUnormUinv;
                    case {'double','single'}
                        rho = 0.5*eps(u)*MZK+0.25*eps(uR)^2*condUnormUinv;
                    case 'half'
                        rho = 0.5*eps(u)*MZK+(2^(-22))*condUnormUinv;
                    case 'mp2'
                        rho = [];
                end
                
               
                
 
    end

    