function varargout = top(varargin)
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @top_OpeningFcn, ...
                   'gui_OutputFcn',  @top_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end

function top_OpeningFcn(hObject, eventdata, handles, varargin)
handles.output = hObject;
guidata(hObject, handles);

function varargout = top_OutputFcn(hObject, eventdata, handles) 
varargout{1} = handles.output;

function edit1_Callback(hObject, eventdata, handles)

function edit1_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function edit2_Callback(hObject, eventdata, handles)

% --- Executes during object creation, after setting all properties.
function edit2_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function edit3_Callback(hObject, eventdata, handles)

% --- Executes during object creation, after setting all properties.
function edit3_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function edit4_Callback(hObject, eventdata, handles)

% --- Executes during object creation, after setting all properties.
function edit4_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function edit5_Callback(hObject, eventdata, handles)

% --- Executes during object creation, after setting all properties.
function edit5_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function point_Callback(hObject, eventdata, handles)


% --- Executes during object creation, after setting all properties.
function point_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
function edit6_Callback(hObject, eventdata, handles)



% --- Executes during object creation, after setting all properties.
function edit6_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes during object creation, after setting all properties.

% --- Executes during object creation, after setting all properties.
function axes_CreateFcn(hObject, eventdata, handles)
imshow('C:\Users\RAJESH\OneDrive\Desktop\FIN\rj.PNG');

% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
base=str2double(get(handles.edit1,'string'));
height=str2double(get(handles.edit2,'string'));
volfrac=str2double(get(handles.edit3,'string'));
nloop=str2double(get(handles.edit4,'string'));
force=str2double(get(handles.edit5,'string'));
rmin =str2double(get(handles.edit6,'string'));
point =get(handles.point,'value');

guidata(hObject,handles);
infill(base,height,volfrac,nloop,point,force,rmin,handles,hObject)
% handles    structure with handles and user data (see GUIDATA)

function infill(base,height,volfrac,nloop,point,force,rmin,handles,hObject)
close all;
%% MATERIAL PROPERTIES
E0 = 1;
Emin = 1e-9;
nu = 0.3;
vol_max = volfrac + 0.1;
penal = 3;      % stiffness penalty
p = 16;         % pNorm
r_hat = 2;      % pNorm radius
% rmin density filter radius
move = 0.01;    % limited move for the design variables
beta = 1;       % beta continuation
eta = 0.5;      % projection threshold, fixed at 0.5
mdof = [1,2];% 1 total volume % 2 upper bound
vol_max_pNorm = (base*height*vol_max^p)^(1/p);
%% PREPARE FINITE ELEMENT ANALYSIS
cellx = 1;
celly = 1;
cellh = 1;
Ke = zeros(8,8);

Ke(1,1) = (8/3 - (8*nu)/3)*cellx^2 + (16*celly^2)/3;
Ke(1,2) = 2*cellx*celly*(nu + 1);
Ke(1,3) = (4/3 - (4*nu)/3)*cellx^2 - (16*celly^2)/3;
Ke(1,4) = 2*cellx*celly*(3*nu - 1);
Ke(1,5) = ((4*nu)/3 - 4/3)*cellx^2 - (8*celly^2)/3;
Ke(1,6) = -2*cellx*celly*(nu + 1);
Ke(1,7) = ((8*nu)/3 - 8/3)*cellx^2 + (8*celly^2)/3;
Ke(1,8) = -2*cellx*celly*(3*nu - 1);

Ke(2,1) = 2*cellx*celly*(nu + 1);
Ke(2,2) = (16*cellx^2)/3 - (8*celly^2*(nu - 1))/3;
Ke(2,3) = -2*cellx*celly*(3*nu - 1);
Ke(2,4) = (8*cellx^2)/3 + (8*celly^2*(nu - 1))/3;
Ke(2,5) = -2*cellx*celly*(nu + 1);
Ke(2,6) = ((4*nu)/3 - 4/3)*celly^2 - (8*cellx^2)/3;
Ke(2,7) = 2*cellx*celly*(3*nu - 1);
Ke(2,8) = (4/3 - (4*nu)/3)*celly^2 - (16*cellx^2)/3;

Ke(3,1) = (4/3 - (4*nu)/3)*cellx^2 - (16*celly^2)/3;
Ke(3,2) = -2*cellx*celly*(3*nu - 1);
Ke(3,3) = (8/3 - (8*nu)/3)*cellx^2 + (16*celly^2)/3;
Ke(3,4) = -2*cellx*celly*(nu + 1);
Ke(3,5) = ((8*nu)/3 - 8/3)*cellx^2 + (8*celly^2)/3;
Ke(3,6) = 2*cellx*celly*(3*nu - 1);
Ke(3,7) = ((4*nu)/3 - 4/3)*cellx^2 - (8*celly^2)/3;
Ke(3,8) = 2*cellx*celly*(nu + 1);

Ke(4,1) = 2*cellx*celly*(3*nu - 1);
Ke(4,2) = (8*cellx^2)/3 + (8*celly^2*(nu - 1))/3;
Ke(4,3) = -2*cellx*celly*(nu + 1);
Ke(4,4) = (16*cellx^2)/3 - (8*celly^2*(nu - 1))/3;
Ke(4,5) = -2*cellx*celly*(3*nu - 1);
Ke(4,6) = (4/3 - (4*nu)/3)*celly^2 - (16*cellx^2)/3;
Ke(4,7) = 2*cellx*celly*(nu + 1);
Ke(4,8) = ((4*nu)/3 - 4/3)*celly^2 - (8*cellx^2)/3;

Ke(5,1) = ((4*nu)/3 - 4/3)*cellx^2 - (8*celly^2)/3;
Ke(5,2) = -2*cellx*celly*(nu + 1);
Ke(5,3) = ((8*nu)/3 - 8/3)*cellx^2 + (8*celly^2)/3;
Ke(5,4) = -2*cellx*celly*(3*nu - 1);
Ke(5,5) = (8/3 - (8*nu)/3)*cellx^2 + (16*celly^2)/3;
Ke(5,6) = 2*cellx*celly*(nu + 1);
Ke(5,7) = (4/3 - (4*nu)/3)*cellx^2 - (16*celly^2)/3;
Ke(5,8) = 2*cellx*celly*(3*nu - 1);

Ke(6,1) = -2*cellx*celly*(nu + 1);
Ke(6,2) = ((4*nu)/3 - 4/3)*celly^2 - (8*cellx^2)/3;
Ke(6,3) = 2*cellx*celly*(3*nu - 1);
Ke(6,4) = (4/3 - (4*nu)/3)*celly^2 - (16*cellx^2)/3;
Ke(6,5) = 2*cellx*celly*(nu + 1);
Ke(6,6) = (16*cellx^2)/3 - (8*celly^2*(nu - 1))/3;
Ke(6,7) = -2*cellx*celly*(3*nu - 1);
Ke(6,8) = (8*cellx^2)/3 + (8*celly^2*(nu - 1))/3;

Ke(7,1) = ((8*nu)/3 - 8/3)*cellx^2 + (8*celly^2)/3;
Ke(7,2) = 2*cellx*celly*(3*nu - 1);
Ke(7,3) = ((4*nu)/3 - 4/3)*cellx^2 - (8*celly^2)/3;
Ke(7,4) = 2*cellx*celly*(nu + 1);
Ke(7,5) = (4/3 - (4*nu)/3)*cellx^2 - (16*celly^2)/3;
Ke(7,6) = -2*cellx*celly*(3*nu - 1);
Ke(7,7) = (8/3 - (8*nu)/3)*cellx^2 + (16*celly^2)/3;
Ke(7,8) = -2*cellx*celly*(nu + 1);

Ke(8,1) = -2*cellx*celly*(3*nu - 1);
Ke(8,2) = (4/3 - (4*nu)/3)*celly^2 - (16*cellx^2)/3;
Ke(8,3) = 2*cellx*celly*(nu + 1);
Ke(8,4) = ((4*nu)/3 - 4/3)*celly^2 - (8*cellx^2)/3;
Ke(8,5) = 2*cellx*celly*(3*nu - 1);
Ke(8,6) = (8*cellx^2)/3 + (8*celly^2*(nu - 1))/3;
Ke(8,7) = -2*cellx*celly*(nu + 1);
Ke(8,8) = (16*cellx^2)/3 - (8*celly^2*(nu - 1))/3;

Kc = (E0/(1-nu^2))*(cellh/(16*cellx*celly))*Ke;

nodenrs = reshape(1:(1+base)*(1+height),1+height,1+base);
edofVec = reshape(2*nodenrs(1:end-1,1:end-1)+1,base*height,1);
edofMat = repmat(edofVec,1,8)+repmat([0 1 2*height+[2 3 0 1] -2 -1],base*height,1);
iK = reshape(kron(edofMat,ones(8,1))',64*base*height,1);
jK = reshape(kron(edofMat,ones(1,8))',64*base*height,1);
% DEFINE LOADS AND SUPPORTS
iLoad = point;
Fsparse = sparse(2*(height+1)*(base+1),1);
if iLoad == 1
    msgbox('Make a Selection in the list');
    
elseif iLoad == 2
    Fsparse(2*(base+1)*(height+1)-(2*height),1) = -force;
    
elseif iLoad == 3 
    Fsparse(2*(height+1)*(base)+height+2,1) = -force;
    
elseif iLoad == 4
    Fsparse(2*(base+1)*(height+1),1) = -force;
    
end
fixeddofs = union([1:1:2*(height+1)],[1]);
U = zeros(2*(height+1)*(base+1),1);
alldofs = [1:2*(height+1)*(base+1)];
freedofs = setdiff(alldofs,fixeddofs);

%% PREPARE FILTER
iH = ones(base*height*(2*(ceil(rmin)-1)+1)^2,1);
jH = ones(size(iH));
sH = zeros(size(iH));
k = 0;
for i1 = 1:base
  for j1 = 1:height
    e1 = (i1-1)*height+j1;
      %%%Finding all neighboring cells whose centers lie within Radius rmin
    for i2 = max(i1-(ceil(rmin)-1),1):min(i1+(ceil(rmin)-1),base)
      for j2 = max(j1-(ceil(rmin)-1),1):min(j1+(ceil(rmin)-1),height)
        e2 = (i2-1)*height+j2;%Neighbors found
          %%%Once neighboring cells found, weights can be assigned to primary & neighboring cells
        k = k+1;
        iH(k) = e1;%Primary cell
        jH(k) = e2;%Its neighboring cells
        sH(k) = max(0,rmin-sqrt((i1-i2)^2+(j1-j2)^2));%Assigning weights based on distance to primary cell
      end
    end
  end
end
H = sparse(iH,jH,sH);
Hs = sum(H,2);

%% PREPARE PDE FILTER
edofVecF = reshape(nodenrs(1:end-1,1:end-1),base*height,1);
edofMatF = repmat(edofVecF,1,4)+repmat([0 height+[1:2] 1],base*height,1);
iKF = reshape(kron(edofMatF,ones(4,1))',16*base*height,1);
jKF = reshape(kron(edofMatF,ones(1,4))',16*base*height,1);
iTF = reshape(edofMatF,4*base*height,1);
jTF = reshape(repmat([1:base*height],4,1)',4*base*height,1);
sTF = repmat(1/4,4*base*height,1);
TF = sparse(iTF,jTF,sTF);

Rmin = r_hat/2/sqrt(3);
KcF = Rmin^2*[4 -1 -2 -1; -1  4 -1 -2; -2 -1  4 -1; -1 -2 -1  4]/6 + ...
             [4  2  1  2;  2  4  2  1;  1  2  4  2;  2  1  2  4]/36;
sKF = reshape(KcF(:)*ones(1,base*height),16*base*height,1);
KF = sparse(iKF,jKF,sKF);
LF = chol(KF,'lower');% factorizes symmetric positive definite matrix KF into an upper triangular R that satisfies KF = R'*R.If KF is nonsymmetric , then chol treats the matrix as symmetric and uses only the diagonal and upper triangle of KF.

%% INITIALIZE ITERATION
x = repmat(volfrac,height,base);%x=phi
xTilde = x;
xPhys = (tanh(beta*eta) + tanh(beta*(xTilde-eta))) / (tanh(beta*eta) + tanh(beta*(1-eta)));
xold1 = reshape(x,[height*base,1]);
xold2 = reshape(x,[height*base,1]);
low = 0;
upp = 0;

loopbeta = 0;
loop = 0;
change = 1;
%% START ITERATION

% store results
c_hist = zeros(nloop,1);        % compliance
vol_hist = zeros(nloop,1);      % volume
change_hist = zeros(nloop,1);   % maximum design change
sharp_hist = zeros(nloop,1);    % sharpness
cons_hist = zeros(nloop,2);     % constraints

while change > 0.0000001 && loop < nloop
    loopbeta = loopbeta+1;
    loop = loop+1;
    %% FE-ANALYSIS
    sK = reshape(Kc(:)*(Emin+xPhys(:)'.^penal*(E0-Emin)),64*base*height,1);
    K = sparse(iK,jK,sK); K = (K+K')/2;%stiffness matrix
    
    U(freedofs) = K(freedofs,freedofs)\Fsparse(freedofs);%displacement

    %% OBJECTIVE FUNCTION AND SENSITIVITY ANALYSIS
    ce = reshape(sum((U(edofMat)*Kc).*U(edofMat),2),height,base);
    c = sum(sum((Emin+xPhys.^penal*(E0-Emin)).*ce));
    dc = -penal*(E0-Emin)*xPhys.^(penal-1).*ce;%-∂c/∂ρe(φTilde) 

    dv = ones(height,base);

    x_pde_hat = (TF'*(LF'\(LF\(TF*xPhys(:)))));
    dfdx_pde = (sum(x_pde_hat.^p))^(1/p-1) * x_pde_hat.^(p-1);
  
    %% FILTERING/MODIFICATION OF SENSITIVITIES
    dx = beta * (1-tanh(beta*(xTilde-eta)).*tanh(beta*(xTilde-eta))) / (tanh(beta*eta) + tanh(beta*(1-eta)));
    dc(:) = H*(dc(:).*dx(:)./Hs);
    dv(:) = H*(dv(:).*dx(:)./Hs);

    %% UPDATE OF DESIGN VARIABLES AND PHYSICAL DENSITIES
    % METHOD OF MOVING ASYMPTOTES (MMA)
    m = size(mdof,2);
    n = base*height;  
    
    df0dx = reshape(dc,[base*height,1]);
    dfdx = zeros(3,base*height);
    dfdx(1,1:base*height) = reshape(dv,[1,base*height])/(base*height*volfrac);
    dfdx(2,1:base*height) = TF'*(LF'\(LF\(TF*dfdx_pde(:))));
    
    ic = 2;
    tmp = reshape(dfdx(ic,:),[height,base]);
    dfdx(ic,:) = reshape(H*(tmp(:).*dx(:)./Hs),[1,base*height]);   
    
    iter = loopbeta;
    xval = reshape(x,[base*height,1]);
    xmin=max(0.0,xval-move);
    xmax=min(1,xval+move);

    f0val = c;
    fval = zeros(2,1);
    fval(1,1) = sum(sum(xPhys)) / (base*height*volfrac) - 1;
    fval(2,1) = (sum(x_pde_hat.^p))^(1/p)- vol_max_pNorm;
    
    a0 = 1;
    a = zeros(m,1);     
    c_ = ones(m,1)*1000;
    d = zeros(m,1);
    [xmma,~,~,~,~,~,~,~,~,low,upp] = ...
        mmasub(m,n,iter,xval,xmin,xmax,xold1,xold2,...
        f0val,df0dx,fval(mdof),dfdx(mdof,:),low,upp,a0,a,c_,d);

    xnew = reshape(xmma,[height,base]);
    xold2 = xold1;
    xold1 = xval;
    
    xTilde(:) = (H*xnew(:))./Hs;
    xPhys = (tanh(beta*eta) + tanh(beta*(xTilde-eta))) / (tanh(beta*eta) + tanh(beta*(1-eta)));

    change = max(abs(xnew(:)-x(:)));
    x = xnew;

    %% PRINT RESULTS
    disp([' It.: ' sprintf('%4i',loop) ' Obj.: ' sprintf('%10.4f',c) ...
        ' Vol.: ' sprintf('%6.3f',sum(sum(xPhys))/(base*height)) ...
        ' Ch.: ' sprintf('%6.3f',change) ...
        ' Cons.: ' sprintf('%6.3f',fval)]);

    %% UPDATE HEAVISIDE REGULARIZATION PARAMETER
    if beta < 100 && (loopbeta >= 40 || change <= 0.001)
        beta = 2*beta;
        loopbeta = 0;
        change = 1;
        fprintf('Parameter beta increased to %g.\n',beta);
    end
    
    %% Store current values
    c_hist(loop,1) = c;
    vol_hist(loop,1) = sum(sum(xPhys))/(base*height);
    change_hist(loop,1) = change;
    cons_hist(loop,:) = fval;
    sharp_hist(loop,1) = 4*sum(sum(xPhys.*(ones(height, base)-xPhys))) / (height*base);

    %% PLOT DENSITIES
    figure(1);
    set(1, 'Position', [100, 450, 540, min(100+540*height/base,540)]);
    colormap(gray); imagesc(-xPhys, [-1 0]); axis equal; axis tight; axis off; drawnow;
    
     
end


function [xmma,ymma,zmma,lam,xsi,eta,mu,zet,s,f0app,fapp] = ...
gcmmasub(m,n,iter,epsimin,xval,xmin,xmax,low,upp, ...
         raa0,raa,f0val,df0dx,fval,dfdx,a0,a,c,d);
%
eeen = ones(n,1);
zeron = zeros(n,1);

% Calculations of the bounds alfa and beta.
albefa = 0.1;
move = 0.5;
%
zzz1 = low + albefa*(xval-low);
zzz2 = xval - move*(xmax-xmin);
zzz  = max(zzz1,zzz2);
alfa = max(zzz,xmin);
zzz1 = upp - albefa*(upp-xval);
zzz2 = xval + move*(xmax-xmin);
zzz  = min(zzz1,zzz2);
beta = min(zzz,xmax);

% Calculations of p0, q0, r0, P, Q, r and b.
xmami = xmax-xmin;
xmamieps = 0.00001*eeen;
xmami = max(xmami,xmamieps);
xmamiinv = eeen./xmami;
ux1 = upp-xval;
ux2 = ux1.*ux1;
xl1 = xval-low;
xl2 = xl1.*xl1;
uxinv = eeen./ux1;
xlinv = eeen./xl1;
%
p0 = zeron;
q0 = zeron;
p0 = max(df0dx,0);
q0 = max(-df0dx,0);
pq0 = p0 + q0;
p0 = p0 + 0.001*pq0;
q0 = q0 + 0.001*pq0;
p0 = p0 + raa0*xmamiinv;
q0 = q0 + raa0*xmamiinv;
p0 = p0.*ux2;
q0 = q0.*xl2;
r0 = f0val - p0'*uxinv - q0'*xlinv;
%
P = sparse(m,n);
Q = sparse(m,n);
P = max(dfdx,0);
Q = max(-dfdx,0);
PQ = P + Q;
P = P + 0.001*PQ;
Q = Q + 0.001*PQ;
P = P + raa*xmamiinv';
Q = Q + raa*xmamiinv';
P = P * spdiags(ux2,0,n,n);
Q = Q * spdiags(xl2,0,n,n);
r = fval - P*uxinv - Q*xlinv;
b = -r;

% Solving the subproblem by a primal-dual Newton method
[xmma,ymma,zmma,lam,xsi,eta,mu,zet,s] = ...
subsolv(m,n,epsimin,low,upp,alfa,beta,p0,q0,P,Q,a0,a,b,c,d);

% Calculations of f0app and fapp.
ux1 = upp-xmma;
xl1 = xmma-low;
uxinv = eeen./ux1;
xlinv = eeen./xl1;
f0app = r0 + p0'*uxinv + q0'*xlinv;
fapp  =  r +   P*uxinv +   Q*xlinv;


