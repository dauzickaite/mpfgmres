% generates data and plots in Carson and Dauzickaite, 
% STABILITY OF SPLIT-PRECONDITIONED FGMRES IN FOUR PRECISIONS
% requires: chop, multi precision NLA kernels, Advanpix
% SuiteSparse files for rajat14, arc130, west0132,fs_183_3

addpath '/chop-master'
addpath '/Multi_precision_NLA_kernels-master'
addpath '/AdvanpixMCT-4.8.5.14607'
addpath '/suite_sparse_problems'



% dense problems
for c=[1:4,6:10]
    genProblemsSolve(c,false,true,[], true, true)

end

c=5; 
genProblemsSolve(c,true,true,[], true, true)

% sparse problems: solve all and plot stuff
genProblemsSolve([],true,false,'rajat14',false,false)
genProblemsSolve([],true,false,'arc130',false,false)
genProblemsSolve([],true,false,'west0132',false,false)
genProblemsSolve([],true,false,'fs_183_3',false,false)