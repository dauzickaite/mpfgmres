function UinvLinvxp = Plu(x,convprc,L,U,P,u,uPr)
mp.Digits(64);

n = size(L,1);

switch uPr
    case 'single'
        x = single(x);
        L = single(L);
        U = single(U);
        P = single(P);
    case 'double'
        x = double(x);
        L = double(L);
        U = double(U);
        P = double(P);
    case 'quad'
        x = mp(x,34);
        L = mp(L,34);
        U = mp(U,34);
        P = mp(P,34);
    case 'exact'
        x = mp(x);
        L = mp(L);
        U = mp(U);
        P = mp(P);
    case 'mp2'
        x = mp(x,2);
        L = mp(L,2);
        U = mp(U,2);
        P = mp(P,2);
end

switch uPr
    case 'half'
        if ~norm(eye(n) - L)
            Linvx = x;
        else
            Linvx = trisol(double(L),double(P*x));
        end
        if ~norm(eye(n) - U)
            UinvLinvx = Linvx;
        else
            UinvLinvx = trisol(double(U),Linvx);
        end
        
    case {'single','double','quad','exact','mp2'}
        if ~norm(eye(n) - L)
            Linvx = x;
        else
            Linvx = L\(P*x);
        end
        
        if ~norm(eye(n) - U)
            UinvLinvx = Linvx;
        else
            UinvLinvx = U\Linvx;
        end
end

if convprc
    switch u 
        case 'single'
            UinvLinvxp = single(UinvLinvx);

        case 'double'
            UinvLinvxp = double(UinvLinvx);
    end
else
    UinvLinvxp = UinvLinvx;
end