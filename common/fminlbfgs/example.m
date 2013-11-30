options = struct('GradObj','on','Display','iter','LargeScale','off','HessUpdate','bfgs','InitialHessType','identity','GoalsExactAchieve',0);
x0 = ones(1,5)*(-pi/2);
tic
[x2,fval2,exitflag,output,grad] = fminlbfgs(@myfun,x0,options);
toc
options = struct('GradObj','on','Display','iter','LargeScale','off','HessUpdate','bfgs','InitialHessType','identity','GoalsExactAchieve',1,'GradConstr',false);
x0 = ones(1,5);
tic
[x2,fval2] = fminlbfgs(@myfun,x0,options);
toc

tic
[x,fval] = fminunc(@myfun,x0,options);
toc
