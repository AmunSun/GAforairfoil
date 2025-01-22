import numpy as np
from .parsec import parsec, yCoord2

def solver(p, uinf, AOA, Npanel):
    """
    Solves for the coefficient of lift of an airfoil using the Vortex panel method
    
    Parameters:
    p: airfoil PARSEC parameters
    uinf: velocity free stream magnitude
    AOA: airfoil angle of attack
    Npanel: Number of panels to solve for
    
    Returns:
    Cl: Coefficient of lift
    maxThickness: Maximum thickness of the airfoil
    """
    # Geometry discretization
    dbeta = np.pi/Npanel
    beta = 0
    Z_u0 = []
    Z_d0 = []
    x0 = []
    
    Uinf = uinf * np.cos(AOA)
    #print("Uinf=",Uinf)
    Vinf = uinf * np.sin(AOA)
    
    # Get PARSEC coefficients
    a = parsec(p)
    
    # Generate coordinates
    for i in np.arange(0, np.pi + dbeta, dbeta):
        x = (1 - np.cos(beta))/2
        x0.append(x)
        zu, zd = yCoord2(a, x)
        Z_u0.append(zu)
        Z_d0.append(zd)
        beta += dbeta
    # Convert lists to numpy arrays
    x0 = np.array(x0)
    #print("x0size=",x0.shape)
    Z_u0 = np.array(Z_u0).reshape(201,)
    #print("Z_u0size=",Z_u0.shape)
    Z_d0 = np.array(Z_d0).reshape(201,)
    #print("Z_d0size=",Z_d0.shape)
    
    maxThickness = np.max(np.abs(Z_u0 - Z_d0))
    Z_d0 = Z_d0[::-1]  # reverse the array
    
    X = np.concatenate([x0[::-1], x0[1:]])
    Y = np.concatenate([Z_d0, Z_u0[1:]])
    #print("Xshape=",X.shape)
    #print("Yshape=",Y.shape)
    dx = np.diff(X,axis=-1)
    #print("dxshape=",dx.shape)
    dz = np.diff(Y,axis=-1)
    #print("dzshape=",dz.shape)
    alpha = np.arctan2(dz, dx)
    
    x_c = dx/2 + X[:-1]  # collocation points
    z_c = dz/2 + Y[:-1]  # collocation points
    X2 = np.sqrt(dx**2 + dz**2)  # Panel length
    
    # Coefficient matrix formation
    n = len(x_c)
    #print("len(xc)=",len(x_c))
    aij = np.zeros((n+1, n+1))
    bij = np.zeros((n, n))
    
    cos_theta = np.cos(alpha)
    #print("costhetasize=",cos_theta.shape)
    sin_theta = np.sin(alpha)
    #rint("sinthetashape=",sin_theta.shape)
    
    for i in range(n):
        Xp = cos_theta * (x_c[i] - X[:-1]) + sin_theta * (z_c[i] - Y[:-1])
        #print("Xpsize=",Xp.shape)
        Yp = -sin_theta * (x_c[i] - X[:-1]) + cos_theta * (z_c[i] - Y[:-1])
        thetaj = np.arctan2(Yp, Xp)
        thetaj1 = np.arctan2(Yp, Xp - X2)
        aij[i, :-1] = -1 / (2 * np.pi) * (thetaj1 - thetaj)
        theta_1 = np.arctan2((z_c[i] - Y[-1]), (x_c[i] - X[-1]))
        theta_2 = np.arctan2(z_c[i] - Y[-1], x_c[i] - 1e7 * X[-1])
        aij[i, -1] = 1 / (2 * np.pi) * (theta_1 - theta_2)
        aij[i, i] = 0.5
        
        R12 = Xp**2 + Yp**2
        R22 = (Xp - X2)**2 + Yp**2
        f = (Xp*np.log(R12) - (Xp - X2)*np.log(R22) + 2*Yp*(thetaj1 - thetaj))
        #print(f"Xp[i]: {Xp[i]}, shape: {np.shape(Xp[i])}")
        #print(f"R12[i]: {R12[i]}, shape: {np.shape(R12[i])}")
        #print("fshape"  ,f.shape)
        #print("bijshape",bij.shape)
        bij[i, :] = 1/(4*np.pi) * f
        bij[i, i] = 1/(2*np.pi) * Xp[i] * np.log(R12[i])
    
    # Right hand side calculation
    nvec = np.vstack((-sin_theta, cos_theta)).reshape(2,n)  # Shape will be (2, n)
    #print("nvecshape=",nvec.shape)
    RHSb = np.array([Uinf, Vinf]).reshape(2, 1) # Shape will be (2, 1)
    RHSb = RHSb.T @ nvec  # This will give shape (1, n)
    #print("RHSbshape=",RHSb.shape)
    # Coefficient matrix
    A = np.zeros((n, n))
    A[:, 0] = aij[:-1, 0] - aij[:-1, -1]
    A[:, -1] = aij[:-1, -2] + aij[:-1, -1]
    A[:, 1:-1] = aij[:-1, 1:-2]
    
    RHS = -bij @ RHSb.T
    gamma = np.linalg.solve(A, RHS)
    
    # Calculate velocity field and pressure coefficients
    #print("Uinfx_cshape=",(Uinf*x_c).shape)
    gamma = gamma.reshape(400,)
    Ueinf = np.zeros(n)
    for i in range(n):
        Ueinf[i] = Uinf * x_c[i] + z_c[i] * Vinf + gamma[i]    # velocity field
    #print("Ueinfshape=",Ueinf.shape)
    Ue = np.zeros(n)
    Cp = np.zeros(n)
    #print("Uesize=",Ue.shape)
    #print("Ueinfshape=",Ueinf.shape)
    #print("X2.shape=",X2.shape)
    for i in range(n-1):
        #print((2*(Ueinf[i] - Ueinf[i+1])/(X2[i] + X2[i+1])).shape)
        Ue[i] = 2*(Ueinf[i] - Ueinf[i+1])/(X2[i] + X2[i+1])
        Cp[i] = 1 - Ue[i]**2/(Uinf**2 + Vinf**2)
    
    # Aerodynamic forces calculation
    print("Y-Yshape=",(Y[1:] - Y[:-1]).shape)
    fx = np.zeros(n)
    fy= np.zeros(n)
    mj = np.zeros(n)
    for j in range(len(Cp)):
        fx[j] = Cp[j] * (Y[j+1] - Y[j])
        fy[j] = Cp[j] * (X[j+1] - X[j])
        mj[j] = -fx[j] * (Y[j+1] + Y[j]) / 2 + fy[j] * ((X[j+1] + X[j]) / 2 - 1/4)
    #fx = Cp * (Y[1:] - Y[:-1])
    #fy = Cp * (X[1:] - X[:-1]) 
    #mj = -fx * (Y[1:-1] + Y[:-2])/2 + fy * ((X[1:-1] + X[:-2])/2 - 1/4)


    #fx = Cp * np.diff(Y[:,:])
    #print("fxshape=",fx.shape)
    #fy = Cp * np.diff(X)
    #mj = -fx * (Y[1:n] + Y[:n-1])/2 + fy * ((X[1:n] + X[:n-1])/2 - 1/4)
    
    Fx = -np.sum(fx)
    Fy = -np.sum(fy)
    M = np.sum(mj)
    
    Cl = -np.sin(AOA)*Fx + np.cos(AOA)*Fy
    Cdp = Fx*np.cos(AOA) + Fy*np.sin(AOA)
    
    return Cl, maxThickness
