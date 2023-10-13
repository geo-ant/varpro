function [alpha, c, wresid, wresid_norm, y_est, Regression] = ...
          varpro(y, w, alpha, n, ada, lb, ub, options)
%VARPRO Solve a separable nonlinear least squares problem.
% [alpha, c, wresid, wresid_norm, y_est, Regression] =
%             VARPRO(y, w, alpha, n, ada, lb, ub, options)
%
% Given a set of m observations y(1),...,y(m)
% this program computes a weighted least squares fit using the model
%
%    eta(alpha,c,t) = 
%            c_1 * phi_1 (alpha,t) + ...  + c_n * phi_n (alpha,t) 
% (possibly with an extra term  + phi_{n+1} (alpha,t) ).
%
% This program determines optimal values of the q nonlinear parameters
% alpha and the n linear parameters c, given observations y at m
% different values of the "time" t and given evaluation of phi and 
% (optionally) derivatives of phi.
%
% On Input:
%
%   y    m x 1   vector containing the m observations.
%   w    m x 1   vector of weights used in the least squares
%                fit.  We minimize the norm of the weighted residual
%                vector r, where, for i=1:m,
%
%                r(i) = w(i) * (y(i) - eta(alpha, c, t(i,:))).
%
%                Therefore, w(i) should be set to 1 divided by
%                the standard deviation in the measurement y(i).  
%                If this number is unknown, set w(i) = 1.
%   alpha q x 1  initial estimates of the parameters alpha.
%                If alpha = [], Varpro assumes that the problem
%                is linear and returns estimates of the c parameters.
%   n            number of linear parameters c
%   ada          a function handle, described below.
%   lb    q x 1  lower bounds on the parameters alpha. 
%   (Optional)   (Omit this argument or use [] if there are
%                no lower bounds.)
%   ub    q x 1  upper bounds on the parameters alpha. 
%   (Optional)   (Omit this argument or use [] if there are
%                no upper bounds.)
%   options      The Matlab optimization parameter structure,
%   (Optional)   set by "optimset", to control convergence
%                tolerances, maximum number of function evaluations,
%                information displayed in the command window, etc. 
%                To use default options, omit this parameter.
%                To determine the default options, type
%                    options = optimset('lsqnonlin')
%                After doing this, the defaults can be modified;
%                to modify the display option, for example, type
%                    options = optimset('lsqnonlin');
%                    optimset(options,'Display','iter');
%
% On Output:
%
%  alpha  q x 1  estimates of the nonlinear parameters.
%  c      n x 1  estimates of the linear parameters.
%  wresid m x 1  weighted residual vector, with i-th component
%                w(i) * (y(i) - eta(alpha, c, t(i,:))).
%  wresid_norm   norm of wresid.
%  y_est  m x 1  the model estimates = eta(alpha, c, t(i,:)))
%  Regression    a structure containing diagnostics about the model fit.
%                **************************************************
%                *                C a u t i o n:                  *
%                *   The theory that makes statistical            *
%                *   diagnostics useful is derived for            *
%                *   linear regression, with no upper- or         *
%                *   lower-bounds on variables.                   *
%                *   The relevance of these quantities to our     *
%                *   nonlinear model is determined by how well    *
%                *   the linearized model (Taylor series model)   *
%                *         eta(alpha_true, c_true)                *
%                *            +  Phi * (c  - c_true)              *
%                *            + dPhi * (alpha - alpha_true)       *
%                *   fits the data in the neighborhood of the     *
%                *   true values for alpha and c, where Phi       *
%                *   and dPhi contain the partial derivatives     *
%                *   of the model with respect to the c and       *
%                *   alpha parameters, respectively, and are      *
%                *   defined in ada.                              *
%                **************************************************
%  Regression.report:  
%                This structure includes information on the solution
%                process, including the number of iterations, 
%                termination criterion, and exitflag from lsqnonlin.
%                (Type 'help lsqnonlin' to see the exit conditions.)
%                Regression.report.rank is the computed rank of the 
%                matrix for the linear subproblem.  If this equals
%                n, then the linear coefficients are well-determined.
%                If it is less than n, then although the model might
%                fit the data well, other linear coefficients might
%                give just as good a fit.
%  Regression.sigma:        
%                The estimate of the standard deviation is the
%                weighted residual norm divided by the square root
%                of the number of degrees of freedom.
%                This is also called the "regression standard error"
%                or the square-root of the weighted SSR (sum squared
%                residual) divided by the square root of the
%                number of degrees of freedom.
%  Regression.RMS:
%                The "residual mean square" is equal to sigma^2:
%                RMS = wresid_norm^2 / (m-n+q)
%  Regression.coef_determ:
%                The "coefficient of determination" for the fit,
%                also called the square of the multiple correlation
%                coefficient, is sometimes called R^2.
%                It is computed as 1 - wresid_norm^2/CTSS,
%                where the "corrected total sum of squares"
%                CTSS is the norm-squared of W*(y-y_bar),
%                and the entries of y_bar are all equal to
%                (the sum of W_i^2 y_i) divided by (the sum of W_i^2).
%                A value of .95, for example, indicates that 95 per 
%                cent of the CTTS is accounted for by the fit.
%
%  Regression.CovMx: (n+q) x (n+q)
%                This is the estimated variance/covariance matrix for
%                the parameters.  The linear parameters c are ordered
%                first, followed by the nonlinear parameters alpha.
%                This is empty if dPhi is not computed by ada.
%  Regression.CorMx: (n+q) x (n+q)
%                This is the estimated correlation matrix for the 
%                parameters.  The linear parameters c are ordered
%                first, followed by the nonlinear parameters alpha.
%                This is empty if dPhi is not computed by ada.
%  Regression.std_param: (n+q) x 1
%                This vector contains the estimate of the standard 
%                deviation for each parameter.
%                The k-th element is the square root of the k-th main 
%                diagonal element of Regression.CovMx
%                This is empty if dPhi is not computed by ada.
%  Regression.t_ratio:   (n+q) x 1
%                The t-ratio for each parameter is equal to the
%                parameter estimate divided by its standard deviation.
%                (linear parameters c first, followed by alpha)
%                This is empty if dPhi is not computed by ada.
%  Regression.standardized_wresid:
%                The k-th component of the "standardized weighted 
%                residual" is the k-th component of the weighted 
%                residual divided by its standard deviation.
%                This is empty if dPhi is not computed by ada.
%
%---------------------------------------------------------------
% Specification of the function ada, which computes information
% related to Phi:
%
%   function [Phi,dPhi,Ind] = ada(alpha)
%
%     This function computes Phi and, if possible, dPhi.
%
%     On Input: 
%
%        alpha q x 1    contains the current value of the alpha parameters.
%
%        Note:  If more input arguments are needed, use the standard
%               Matlab syntax to accomplish this.  For example, if
%               the input arguments to ada are t, z, and alpha, then
%               before calling varpro, initialize t and z, and in calling 
%               varpro, replace "@ada" by "@(alpha)ada(t,z,alpha)".
%
%     On Output:
%
%        Phi   m x n1   where Phi(i,j) = phi_j(alpha,t_i).
%                       (n1 = n if there is no extra term; 
%                        n1 = n+1 if an extra term is used)
%        dPhi  m x p    where the columns contain partial derivative
%                       information for Phi and p is the number of 
%                       columns in Ind 
%                       (or dPhi = [] if derivatives are not available).
%        Ind   2 x p    Column k of dPhi contains the partial
%                       derivative of Phi_j with respect to alpha_i, 
%                       evaluated at the current value of alpha, 
%                       where j = Ind(1,k) and i = Ind(2,k).
%                       Columns of dPhi that are always zero, independent
%                       of alpha, need not be stored. 
%        Example:  if  phi_1 is a function of alpha_2 and alpha_3, 
%                  and phi_2 is a function of alpha_1 and alpha_2, then 
%                  we can set
%                          Ind = [ 1 1 2 2
%                                  2 3 1 2 ]
%                  In this case, the p=4 columns of dPhi contain
%                          d phi_1 / d alpha_2,
%                          d phi_1 / d alpha_3,
%                          d phi_2 / d alpha_1,
%                          d phi_2 / d alpha_2,
%                  evaluated at each t_i.
%                  There are no restrictions on how the columns of
%                  dPhi are ordered, as long as Ind correctly specifies
%                  the ordering.
%
%        If derivatives dPhi are not available, then set dPhi = Ind = [].
%      
%---------------------------------------------------------------
%
%  Varpro calls lsqnonlin, which solves a constrained least squares
%  problem with upper and lower bounds on the constraints.  What
%  distinguishes varpro from lsqnonlin is that, for efficiency and
%  reliability, varpro causes lsqnonlin to only iterate on the
%  nonlinear parameters.  Given the information in Phi and dPhi, this 
%  requires an intricate but inexpensive computation of partial 
%  derivatives, and this is handled by the varpro function formJacobian.
%
%  lsqnonlin is in Matlab's Optimization Toolbox.  Another solver
%  can be substituted if the toolbox is not available.
%
%  Any parameters that require upper or lower bounds should be put in
%  alpha, not c, even if they appear linearly in the model.
%  
%  The original Fortran implementation of the variable projection 
%  algorithm (ref. 2) was modified in 1977 by John Bolstad 
%  Computer Science Department, Serra House, Stanford University,
%  using ideas of Linda Kaufman (ref. 5) to speed up the 
%  computation of derivatives.  He also allowed weights on
%  the observations, and computed the covariance matrix.
%  Our Matlab version takes advantage of 30 years of improvements
%  in programming languages and minimization algorithms.  
%  In this version, we also allow upper and lower bounds on the 
%  nonlinear parameters.
%
%  It is hoped that this implementation will be of use to Matlab
%  users, but also that its simplicity will make it easier for the
%  algorithm to be implemented in other languages.
%
%  This program is documented in
%  Dianne P. O'Leary and Bert W. Rust,
%  Variable Projection for Nonlinear Least Squares Problems,
%  US National Inst. of Standards and Technology, 2010.
%
%  Main reference:
%
%    0.  Gene H. Golub and V. Pereyra, 'Separable nonlinear least   
%        squares: the variable projection method and its applications,'
%        Inverse Problems 19, R1-R26 (2003).
%         
%  See also these papers, cited in John Bolstad's Fortran code:
%                                                                       
%    1.  Gene H. Golub and V. Pereyra, 'The differentiation of      
%        pseudo-inverses and nonlinear least squares problems whose 
%        variables separate,' SIAM J. Numer. Anal. 10, 413-432      
%         (1973).                                                    
%    2.  ------, same title, Stanford C.S. Report 72-261, Feb. 1972.
%    3.  Michael R. Osborne, 'Some aspects of non-linear least      
%        squares calculations,' in Lootsma, Ed., 'Numerical Methods 
%        for Non-Linear Optimization,' Academic Press, London, 1972.
%    4.  Fred Krogh, 'Efficient implementation of a variable projection
%        algorithm for nonlinear least squares problems,'           
%        Comm. ACM 17:3, 167-169 (March, 1974).                    
%    5.  Linda Kaufman, 'A variable projection method for solving  
%        separable nonlinear least squares problems', B.I.T. 15,   
%        49-57 (1975).                                             
%    6.  C. Lawson and R. Hanson, Solving Least Squares Problems,
%        Prentice-Hall, Englewood Cliffs, N. J., 1974.          
%
%  These books discuss the statistical background:
%
%    7.  David A. Belsley, Edwin Kuh, and Roy E. Welsch, Regression 
%        Diagnostics, John Wiley & Sons, New York, 1980, Chap. 2.
%    8.  G.A.F. Seber and C.J. Wild, Nonlinear Regression,
%        John Wiley & Sons, New York, 1989, Sec. 2.1, 5.1, and 5.2.
%
%  Dianne P. O'Leary, NIST and University of Maryland, February 2011.
%  Bert W. Rust,      NIST                             February 2011.
%  Comments updated 07-2011.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

global mydebug myneglect   % test neglect of extra term in Jacobian

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Initialization: Check input, set default parameters and options.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% special case for octave
is_octave = exist('OCTAVE_VERSION', 'builtin') > 0;
if (is_octave)
  pkg load optim;
end

[m,ell] = size(y);      % m = number of observations
[m1,ell1] = size(w);

if (m1 ~= m) | (ell > 1) | (ell1 > 1)
  error('y and w must be column vectors of the same length')
end

[q,ell] = size(alpha);   % q = number of nonlinear parameters

if (ell > 1)
  error('alpha must be a column vector containing initial guesses for nonlinear parameters')
end

if (nargin < 6)  
  lb = [];
else
  [q1,ell] = size(lb);
  if (q1 > 0) & (ell > 0)
    if (q1 ~= q) | (ell > 1)
      error('lb must be empty or a column vector of the same length as alpha')
    end
  end
end

if (nargin < 7)
  ub = [];
else
  [q1,ell] = size(ub);
  if (q1 > 0) & (ell > 0)
    if (q1 ~= q) | (ell > 1)
      error('ub must be empty or a column vector of the same length as alpha')
    end
  end
end

if (nargin < 8)
  options = optimset('lsqnonlin');  
end

if (~strcmp(options.Display,'off'))
  disp(sprintf('\n-------------------'))
  disp(sprintf('VARPRO is beginning.'))
end

W = spdiags(w,0,m,m);  % Create an m x m diagonal matrix from the vector w

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Make the first call to ada and do some error checking.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[Phi, dPhi, Ind] = feval(ada, alpha);

[m1,n1] = size(Phi);     % n1 = number of basis functions Phi.

[m2,n2] = size(dPhi);
[ell,n3] = size(Ind);

if (m1 ~= m2) & (m2 > 0)
  error('In user function ada: Phi and dPhi must have the same number of rows.')
end

if (n1 < n) | (n1 > n+1)
  error('In user function ada: The number of columns in Phi must be n or n+1.')
end

if (n2 > 0) & (ell ~= 2)
  error('In user function ada: Ind must have two rows.')
end

if (n2 > 0) & (n2 ~= n3)
  error('In user function ada: dPhi and Ind must have the same number of columns.')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Solve the least squares problem using lsqnonlin or, if there
% are no nonlinear parameters, using the SVD procedure in formJacobian.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if (q > 0) % The problem is nonlinear.

  if ~isempty(dPhi)
    options = optimset(options,'Jacobian','on');
  end

  [alpha, wresid_norm2, wresid, exitflag,output] = ...
    lsqnonlin(@f_lsq, alpha, lb, ub, options);
  [r, Jacobian, Phi, dPhi, y_est, rank] = f_lsq(alpha);
  wresid_norm = sqrt(wresid_norm2);
  Regression.report = output;
  Regression.report.rank = rank;
  Regression.report.exitflag = exitflag;

else       % The problem is linear.

  if (~strcmp(options.Display,'off'))
    disp(sprintf('VARPRO problem is linear, since length(alpha)=0.'))
  end

  [Jacobian, c, wresid, y_est, Regression.report.rank] =  ...
    formJacobian(alpha, Phi, dPhi);
  wresid_norm = norm(wresid);
  wresid_norm2 = wresid_norm * wresid_norm;

end % if q > 0

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  Compute some statistical diagnostics for the solution.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Calculate sample variance,  the norm-squared of the residual
%    divided by the number of degrees of freedom.

sigma2 = wresid_norm2 / (m-n-q);

% Compute  Regression.sigma:        
%                square-root of weighted residual norm squared divided 
%                by number of degrees of freedom.

Regression.sigma = sqrt(sigma2);

% Compute  Regression.coef_determ:
%                The coeficient of determination for the fit,
%                also called the square of the multiple correlation
%                coefficient, or R^2.
%                It is computed as 1 - wresid_norm^2/CTSS,
%                where the "corrected total sum of squares"
%                CTSS is the norm-squared of W*(y-y_bar),
%                and the entries of y_bar are all equal to
%                (the sum of W_i y_i) divided by (the sum of W_i).

y_bar = sum(w.*y) / sum(w);
CTTS = norm(W * (y - y_bar)) ^2;
Regression.coef_determ = 1 - wresid_norm^2 / CTTS;

% Compute  Regression.RMS = sigma^2:
%                the weighted residual norm divided by the number of
%                degrees of freedom.
%                RMS = wresid_norm / sqrt(m-n+q)

Regression.RMS = sigma2;                        

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  Compute some additional statistical diagnostics for the 
%  solution, if the user requested it.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if (nargout==6) 

  if (isempty(dPhi))

    Regression.CovMx = [];
    Regression.CorMx = [];
    Regression.std_param = [];
    Regression.t_ratio = [];
    Regression.standardized_wresid = [];

  else

    % Calculate the covariance matrix CovMx, which is sigma^2 times the
    % inverse of H'*H, where  
    %              H = W*[Phi,J] 
    % contains the partial derivatives of  wresid  with
    % respect to the parameters in alpha and c.

    [xx,pp] = size(dPhi);
    J = zeros(m,q);
    for kk = 1:pp,
      j = Ind(1,kk);
      i = Ind(2,kk);
      if (j > n)
        J(:,i) = J(:,i) + dPhi(:,kk);
      else
        J(:,i) = J(:,i) + c(j) * dPhi(:,kk);
      end
    end
    [Qj,Rj,Pj] = qr(W*[Phi(:,1:n),J], 0);     % Uses compact pivoted QR.
    T2 = Rj \ (eye(size(Rj,1)));
    CovMx = sigma2 * T2 * T2';
    Regression.CovMx(Pj,Pj) = CovMx;  % Undo the pivoting permutation.

    % Compute  Regression.CorMx:        
    %                estimated correlation matrix (n+q) x (n+q) for the
    %                parameters.  The linear parameters are ordered
    %                first, followed by the nonlinear parameters.

    d = 1 ./ sqrt(diag(Regression.CovMx));
    D = spdiags(d,0,n+q,n+q);
    Regression.CorMx = D * Regression.CovMx * D;

    % Compute  Regression.std_param:
    %                The k-th element is the square root of the k-th main 
    %                diagonal element of CovMx. 

    Regression.std_param = sqrt(diag(Regression.CovMx));

    % Compute  Regression.t_ratio:
    %                parameter estimates divided by their standard deviations.

    alpha = reshape(alpha, q, 1);
    Regression.t_ratio = [c; alpha] .* d;

    % Compute  Regression.standardized_wresid:
    %                The k-th component is the k-th component of the
    %                weighted residual, divided by its standard deviation.
    %                Let X = W*[Phi,J], 
    %                    h(k) = k-th main diagonal element of covariance
    %                           matrix for wresid
    %                         = k-th main diagonal element of X*inv(X'*X)*X' 
    %                         = k-th main diagonal element of Qj*Qj'.
    %                Then the standard deviation is estimated by
    %                sigma*sqrt(1-h(k)).

    for k=1:m
      temp(k,1) = Qj(k,:) * Qj(k,:)';
    end
    Regression.standardized_wresid = wresid ./(Regression.sigma*sqrt(1-temp));

  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% End of statistical diagnostics computations.
% Print some final information if desired.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


if (~strcmp(options.Display,'off'))
  disp(sprintf(' '))
  disp(sprintf('VARPRO Results:'))
  disp(sprintf(' Linear Parameters:'))
  disp(sprintf(' %15.7e ',c))

  disp(sprintf(' Nonlinear Parameters:'))
  disp(sprintf(' %15.7e ',alpha))
  disp(sprintf(' '))

  disp(sprintf(' Norm-squared of weighted residual  = %15.7e',wresid_norm2))
  disp(sprintf(' Norm-squared of data vector        = %15.7e',norm(w.*y)^2))
  disp(sprintf(' Norm         of weighted residual  = %15.7e',wresid_norm))
  disp(sprintf(' Norm         of data vector        = %15.7e',norm(w.*y)))
  disp(sprintf(' Expected error of observations     = %15.7e',Regression.sigma))
  disp(sprintf(' Coefficient of determination       = %15.7e',Regression.coef_determ))
  disp(sprintf('VARPRO is finished.'))
  disp(sprintf('-------------------\n'))
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% The computation is now completed.  
%
% varpro uses the following two functions, f_lsq and formJacobian.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%--------------------------- Beginning of f_lsq --------------------------

  function [wr_trial, Jacobian, Phi_trial, dPhi_trial, y_est,myrank] = ...
      f_lsq(alpha_trial)

    % function [wr_trial,Jacobian,Phi_trial,dPhi_trial,y_est,myrank] = ...
    %                                                       f_lsq(alpha_trial)
    %
    % This function is used by lsqnonlin to compute 
    %          wr_trial   the current weighted residual
    %          Jacobian   the Jacobian matrix for the nonlinear parameters
    % It also computes
    %          Phi_trial  the current Phi matrix
    %          dPhi_trial the partial derivatives of Phi_trial (if available).
    %          y_est      the model estimates of y
    %          myrank     the rank of the matrix W*Phi in the linear problem.
    %
    % It uses the user-supplied function ada and the Varpro function formJacobian.

    [Phi_trial, dPhi_trial, Ind] = feval(ada, alpha_trial);

    [Jacobian, c, wr_trial, y_est,myrank] = ...
      formJacobian(alpha_trial, Phi_trial, dPhi_trial);

  end %--------------------------- End of f_lsq ---------------------------

  %----------------------- Beginning of formJacobian ----------------------

  function [Jacobian, c, wresid, y_est,myrank] = formJacobian(alpha, Phi, dPhi)

    % function [Jacobian, c, wresid, y_est,myrank] =  formJacobian(alpha, Phi, dPhi)
    %
    % This function computes the Jacobian, the linear parameters c,
    % and the residual for the model with nonlinear parameters alpha.
    % It is used by the functions Varpro and f_lsq.
    %
    % Notation: there are m data observations
    %                     n1 basis functions (columns of Phi)
    %                     n linear parameters c
    %                       (n = n1, or n = n1 - 1)
    %                     q nonlinear parameters alpha
    %                     p nonzero columns of partial derivatives of Phi 
    %
    % Input:
    %
    %      alpha  q x 1   the nonlinear parameters,
    %      Phi    m x n1  the basis functions Phi(alpha),
    %      dPhi   m x p   the partial derivatives of Phi
    %
    % The variables W, y, q, m, n1, and n are also used.
    %
    % Output: 
    %
    %      Jacobian  m x p the Jacobian matrix, with J(i,k) = partial
    %                      derivative of W resid(i) with respect to alpha(k).
    %      c      n x 1 the optimal linear parameters for this choice of alpha.
    %      wresid    m x 1 the weighted residual = W(y - Phi * c)
    %      y_est     m x 1 the model estimates = Phi * c)
    %      myrank    1 x 1 the rank of the matrix W*Phi.


    % First we determine the optimal linear parameters c for
    % the given values of alpha, and the resulting residual.
    %
    % We use the singular value decomposition to solve the linear least
    % squares problem
    %
    %    min_{c} || W resid ||.
    %       resid =  y - Phi * c.
    %
    % If W*Phi has any singular value less than m * its largest singular value, 
    % these singular values are set to zero.

    [U,S,V] = svd(W*Phi(:,1:n));

    % Three cases: Usually n > 1, but n = 1 and n = 0 require
    % special handling (one or no linear parameters).

    if (n > 1)
      s = diag(S);
    elseif (n==1)
      s = S;
    else    %  n = 0
      if isempty(Ind)
        Jacobian = [];
      else
        Jacobian = zeros(length(y),length(alpha));
        Jacobian(:,Ind(2,:)) = -W*dPhi;
      end
      c = [];
      y_est = Phi;
      wresid = W * (y - y_est);
      myrank = 1;
      return
    end

    tol = m * eps;
    myrank = sum(s > tol*s(1) ); % number of singular values > tol*norm(W*Phi)
    s = s(1:myrank);

    if (myrank < n) & (~strcmp(options.Display,'off'))
      disp(sprintf('Warning from VARPRO:'))
      disp(sprintf('   The linear parameters are currently not well-determined.'))
      disp(sprintf('   The rank of the matrix in the subproblem is %d',myrank))
      disp(sprintf('   which is less than the n=%d linear parameters.',n))
    end

    yuse = y;
    if (n < n1)
      yuse  = y - Phi(:,n1); % extra function Phi(:,n+1)
    end
    temp  = U(:,1:myrank)' * (W*yuse);    
    c = V(:,1:myrank) * (temp./s);
    y_est = Phi(:,1:n) * c;
    wresid = W * (yuse - y_est);
    if (n < n1)
      y_est = y_est + Phi(:,n1);
    end

    if (q == 0) | (isempty(dPhi))
      Jacobian = [];
      return
    end

    % Second, we compute the Jacobian.
    % There are two pieces, which we call Jac1 and Jac2,
    % with Jacobian = - (Jac1 + Jac2).
    %
    % The formula for Jac1 is (P D(W*Phi) pinv(W*Phi) y,
    %             and Jac2 is ((W*Phi)^+})^T (P D(W*Phi))^T y.
    % where  P           is the projection onto the orthogonal complement
    %                       of the range of W*Phi,
    %        D(W*Phi)    is the m x n x q tensor of derivatives of W*Phi,
    %        pinv(W*Phi) is the pseudo-inverse of W*Phi.
    %  (See Golub&Pereyra (1973) equation (5.4).  We use their notational  
    %   conventions for multiplications by tensors.)
    %
    % Golub&Pereyra (2003), p. R5 break these formulas down by columns:
    %     The j-th column of Jac1 is P D_j pinv(W*Phi) y
    %                             =  P D_j c             
    % and the j-th column of Jac2 is (P D_j pinv(W*Phi))^T y,
    %                             =  (pinv(W*Phi))^T D_j^T P^T y
    %                             =  (pinv(W*Phi))^T D_j^T wresid.
    % where D_j is the m x n matrix of partial derivatives of W*Phi
    %     with respect to alpha(j).

    % We begin the computation by precomputing 
    %       WdPhi, which contains the derivatives of W*Phi, and 
    %       WdPhi_r, which contains WdPhi' * wresid.

    WdPhi = W * dPhi;
    WdPhi_r = WdPhi' * wresid;
    T2 = zeros(n1, q);
    ctemp = c;
    if (n1 > n)
      ctemp = [ctemp; 1];
    end

    %   Now we work column-by-column, for j=1:q.
    %
    %   We form Jac1 = D(W*Phi) ctemp.
    %   After the loop, this matrix is multiplied by 
    %        P = U(:,myrank+1:m)*(U(:,myrank+1:m)'
    %   to complete the computation.
    % 
    %   We also form  T2 = (D_j(W*Phi))^T wresid  by unpacking
    %   the information in WdPhi_r, using Ind.
    %   After the loop, T2 is multiplied by the pseudoinverse
    %       (pinv(W*Phi))^T = U(:,1:myrank) * diag(1./s) * (V(:,1:myrank)'
    %   to complete the computation of Jac2.
    %   Note: if n1 > n, last row of T2 is not needed.   

    for j=1:q,                            % for each nonlinear parameter alpha(j)
      range = find(Ind(2,:)==j);        % columns of WdPhi relevant to alpha(j)
      indrows = Ind(1,range);           % relevant rows of ctemp
      Jac1(:,j) = WdPhi(:,range) * ctemp(indrows);      
      T2(indrows,j) = WdPhi_r(range);
    end


    Jac1 = U(:,myrank+1:m) * (U(:,myrank+1:m)' * Jac1);

    T2 = diag(1 ./ s(1:myrank)) * (V(:,1:myrank)' * T2(1:n,:));
    Jac2 = U(:,1:myrank) * T2;

    Jacobian = -(Jac1 + Jac2);

    if (mydebug)
      disp(sprintf('VARPRO norm(neglected Jacobian)/norm(Jacobian) = %e',...
        norm(Jac2,'fro')/norm(Jacobian,'fro') ))
      if (myneglect)
        disp('neglecting')
        Jacobian = -Jac1;
      end
    end

  end %-------------------------- End of formJacobian ----------------------

end %------------------------------ End of varpro ------------------------

