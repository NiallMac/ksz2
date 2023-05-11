import pixell.curvedsky as cs
import numpy as np
from orphics import mpi
from pixell import utils
from pixell.mpi import FakeCommunicator

def four_split_K(qfunc, Xdat_0,Xdat_1,Xdat_2,Xdat_3,Xdatp_0=None,Xdatp_1=None,Xdatp_2=None,Xdatp_3=None):
    """Return kappa_alms combinations required for the 4cross estimator.
    Args:
        Xdat_0 (array): [fTalm,fEalm,fBalm] list of filtered alms from split 0
        Xdat_1 (array): [fTalm,fEalm,fBalm] list of filtered alms from split 1
        Xdat_2 (array): [fTalm,fEalm,fBalm] list of filtered alms from split 2
        Xdat_3 (array): [fTalm,fEalm,fBalm] list of filtered alms from split 3
        q_func1 (function): function for quadratic estimator
        Xdatp_0 (array): [fTalm,fEalm,fBalm] list of filtered alms from split 0 used for RDN0 for different sim data combination
        Xdatp_1 (array): [fTalm,fEalm,fBalm] list of filtered alms from split 1 used for RDN0 for different sim data combination
        Xdatp_2 (array): [fTalm,fEalm,fBalm] list of filtered alms from split 2 used for RDN0 for different sim data combination
        Xdatp_3 (array): [fTalm,fEalm,fBalm] list of filtered alms from split 3 used for RDN0 for different sim data combination
        qfunc2 ([type], optional): [description]. Defaults to None.
    Returns:
        array: Combination of reconstructed kappa alms
    """
    if Xdatp_0 is None:        
        phi_xy00 = (qfunc(Xdat_0,Xdat_0))
        phi_xy11 = (qfunc(Xdat_1,Xdat_1))
        phi_xy22 = (qfunc(Xdat_2,Xdat_2))
        phi_xy33 = (qfunc(Xdat_3,Xdat_3))
        phi_xy01 = 0.5*((qfunc(Xdat_0,Xdat_1))+(qfunc(Xdat_1,Xdat_0)))
        phi_xy02 = 0.5*((qfunc(Xdat_0,Xdat_2))+(qfunc(Xdat_2,Xdat_0)))
        phi_xy03 = 0.5*((qfunc(Xdat_0,Xdat_3))+(qfunc(Xdat_3,Xdat_0)))
        phi_xy10=phi_xy01
        phi_xy12= 0.5*((qfunc(Xdat_1,Xdat_2))+(qfunc(Xdat_2,Xdat_1)))
        phi_xy13= 0.5*((qfunc(Xdat_1,Xdat_3))+(qfunc(Xdat_3,Xdat_1)))
        phi_xy20=phi_xy02
        phi_xy21=phi_xy12
        phi_xy23=0.5*((qfunc(Xdat_2,Xdat_3))+(qfunc(Xdat_3,Xdat_2)))
        phi_xy30=phi_xy03
        phi_xy31=phi_xy13
        phi_xy32=phi_xy23
        phi_xy_hat=(phi_xy00+phi_xy11+phi_xy22+phi_xy33+phi_xy01+phi_xy02+phi_xy03+phi_xy10+phi_xy12+phi_xy13+phi_xy20+phi_xy21+phi_xy23+phi_xy30+phi_xy31+phi_xy32)/4**2
        phi_xy_X=phi_xy_hat-(phi_xy00+phi_xy11+phi_xy22+phi_xy33)/4**2                        
        phi_xy0=(phi_xy00+phi_xy01+phi_xy02+phi_xy03)/4
        phi_xy1=(phi_xy10+phi_xy11+phi_xy12+phi_xy13)/4
        phi_xy2=(phi_xy20+phi_xy21+phi_xy22+phi_xy23)/4
        phi_xy3=(phi_xy30+phi_xy31+phi_xy32+phi_xy33)/4
        phi_xy_x0=phi_xy0-phi_xy00/4
        phi_xy_x1=phi_xy1-phi_xy11/4
        phi_xy_x2=phi_xy2-phi_xy22/4
        phi_xy_x3=phi_xy3-phi_xy33/4
    
    else:
       
        phi_xy00 = (qfunc(Xdat_0,Xdatp_0))
        phi_xy11 = (qfunc(Xdat_1,Xdatp_1))
        phi_xy22 = (qfunc(Xdat_2,Xdatp_2))
        phi_xy33 = (qfunc(Xdat_3,Xdatp_3))
        phi_xy01 = 0.5*((qfunc(Xdat_0,Xdatp_1))+(qfunc(Xdat_1,Xdatp_0)))
        phi_xy02 = 0.5*((qfunc(Xdat_0,Xdatp_2))+(qfunc(Xdat_2,Xdatp_0)))
        phi_xy03 = 0.5*((qfunc(Xdat_0,Xdatp_3))+(qfunc(Xdat_3,Xdatp_0)))
        phi_xy10=phi_xy01
        phi_xy12= 0.5*((qfunc(Xdat_1,Xdatp_2))+(qfunc(Xdat_2,Xdatp_1)))
        phi_xy13= 0.5*((qfunc(Xdat_1,Xdatp_3))+(qfunc(Xdat_3,Xdatp_1)))
        phi_xy20=phi_xy02
        phi_xy21=phi_xy12
        phi_xy23=0.5*((qfunc(Xdat_2,Xdatp_3))+(qfunc(Xdat_3,Xdatp_2)))
        phi_xy30=phi_xy03
        phi_xy31=phi_xy13
        phi_xy32=phi_xy23
        phi_xy_hat=(phi_xy00+phi_xy11+phi_xy22+phi_xy33+phi_xy01+phi_xy02+phi_xy03+phi_xy10+phi_xy12+phi_xy13+phi_xy20+phi_xy21+phi_xy23+phi_xy30+phi_xy31+phi_xy32)/4**2
        phi_xy_X=phi_xy_hat-(phi_xy00+phi_xy11+phi_xy22+phi_xy33)/4**2                        
        phi_xy0=(phi_xy00+phi_xy01+phi_xy02+phi_xy03)/4
        phi_xy1=(phi_xy10+phi_xy11+phi_xy12+phi_xy13)/4
        phi_xy2=(phi_xy20+phi_xy21+phi_xy22+phi_xy23)/4
        phi_xy3=(phi_xy30+phi_xy31+phi_xy32+phi_xy33)/4
        phi_xy_x0=phi_xy0-phi_xy00/4
        phi_xy_x1=phi_xy1-phi_xy11/4
        phi_xy_x2=phi_xy2-phi_xy22/4
        phi_xy_x3=phi_xy3-phi_xy33/4

    phi_xy=np.array([phi_xy_X,phi_xy01,phi_xy02,phi_xy03,phi_xy12,phi_xy13,phi_xy23,phi_xy_x0,phi_xy_x1,phi_xy_x2,phi_xy_x3])
    

    return phi_xy

def split_phi_to_cl(xy,uv,m=4,cross=False,ikalm=None):
    """Obtain 4cross spectra from combinations of kappa_alms build from splits
    Args:
        xy (array): kappa_alms obained from four_split_phi
        uv (array): kappa_alms obained from four_split_phi, usually same as xy except for RDN0.
        m (int, optional): Number of splits. Defaults to 4.
        cross (bool, optional): Set to true to compute the cross power spectrum. Defaults to False.
        ikalm (array, optional): Input kappa_alms used for cross power spectrum. Defaults to None.
    Returns:
        array: clkk
    """
    phi_x=xy[0];phi01=xy[1];phi02=xy[2];phi03=xy[3];phi12=xy[4];phi13=xy[5];phi23=xy[6];phi_x0=xy[7];phi_x1=xy[8];phi_x2=xy[9];phi_x3=xy[10]
    phi_xp=uv[0];phi01p=uv[1];phi02p=uv[2];phi03p=uv[3];phi12p=uv[4];phi13p=uv[5];phi23p=uv[6];phi_x0p=uv[7];phi_x1p=uv[8];phi_x2p=uv[9];phi_x3p=uv[10]
    if cross is False:
        tg1=m**4*cs.alm2cl(phi_x,phi_xp)
        tg2=-4*m**2*(cs.alm2cl(phi_x0,phi_x0p)+cs.alm2cl(phi_x1,phi_x1p)+cs.alm2cl(phi_x2,phi_x2p)+cs.alm2cl(phi_x3,phi_x3p))
        tg3=m*(cs.alm2cl(phi01,phi01p)+cs.alm2cl(phi02,phi02p)+cs.alm2cl(phi03,phi03p)+cs.alm2cl(phi12,phi12p)+cs.alm2cl(phi13,phi13p)+cs.alm2cl(phi23,phi23p))
    else:
        tg1=m**4*cs.alm2cl(phi_x,ikalm)
        tg2=-4*m**2*(cs.alm2cl(phi_x0,ikalm)+cs.alm2cl(phi_x1,ikalm)+cs.alm2cl(phi_x2,ikalm)+cs.alm2cl(phi_x3,ikalm))
        tg3=m*(cs.alm2cl(phi01,ikalm)+cs.alm2cl(phi02,ikalm)+cs.alm2cl(phi03,ikalm)+cs.alm2cl(phi12,ikalm)+cs.alm2cl(phi13,ikalm)+cs.alm2cl(phi23,ikalm))

    auto =(1/(m*(m-1)*(m-2)*(m-3)))*(tg1+tg2+tg3)
    return auto

def mcrdn0_s4(nsims, power, qfunc_AB, split_K_func,
              get_sim_alms_A, data_split_alms_A,
              get_sim_alms_B=None, get_sim_alms_C=None, get_sim_alms_D=None,
              data_split_alms_B=None, data_split_alms_C=None,data_split_alms_D=None,
              qfunc_CD=None,
              use_mpi=True, verbose=True, skip_rd=False, power_mcn0=None):

    """
    get_sim_alms_<X> return list of alms, 1 for each split
    """
    
    if qfunc_CD is None:
        qfunc_CD = qfunc_AB

    mcn0evals = []
    if not(skip_rd): 
        assert data_split_alms_A is not None # Data
        if data_split_alms_B is None:
            data_split_alms_B = data_split_alms_A
        if data_split_alms_C is None:
            data_split_alms_C = data_split_alms_A
        if data_split_alms_D is None:
            data_split_alms_D = data_split_alms_A
        rdn0evals = []

    if use_mpi:
        comm,rank,my_tasks = mpi.distribute(nsims, allow_empty=True)
    else:
        comm,rank,my_tasks = FakeCommunicator(), 0, range(nsims)
        

    for i in my_tasks:
        if verbose:
            print("MCRDN0: Rank %d doing task %d" % (rank,i))

        #get the sim alms
        def get_sim_alms(seed):
            sim_alms_A = get_sim_alms_A(seed)
            if get_sim_alms_B is not None:
                sim_alms_B = get_sim_alms_B(seed)
            else:
                sim_alms_B = sim_alms_A
            if get_sim_alms_C is not None:
                sim_alms_C = get_sim_alms_C(seed)
            else:
                sim_alms_C = sim_alms_A
            if get_sim_alms_D is not None:
                sim_alms_D = get_sim_alms_D(seed)
            else:
                sim_alms_D = sim_alms_A
            return sim_alms_A, sim_alms_B, sim_alms_C, sim_alms_D

        if verbose:
            print("getting sim alms")
        sim_alms_A, sim_alms_B, sim_alms_C, sim_alms_D = get_sim_alms(2*i)
        sim_alms_Ap, sim_alms_Bp, sim_alms_Cp, sim_alms_Dp = get_sim_alms(2*i+1)
        
        #Xs0  = get_kmap((icov,0,i))
        #Xs1= get_kmap1((icov,0,i))
        #Xs2= get_kmap2((icov,0,i))
        #Xs3= get_kmap3((icov,0,i))


        if not(skip_rd):
            #replaces qaXXs
            if verbose:
                print("doing DS terms")
            qABs = split_K_func(qfunc_AB,
                                data_split_alms_A[0], data_split_alms_A[1], data_split_alms_A[2], data_split_alms_A[3],
                                sim_alms_B[0], sim_alms_B[1], sim_alms_B[2], sim_alms_B[3])
            #replaces qbXXs
            qCDs = split_K_func(qfunc_CD,
                                data_split_alms_C[0], data_split_alms_C[1], data_split_alms_C[2], data_split_alms_C[3],
                                sim_alms_D[0], sim_alms_D[1], sim_alms_D[2], sim_alms_D[3])
            #replaces qaXsX
            qAsB = split_K_func(qfunc_AB,
                                sim_alms_A[0], sim_alms_A[1], sim_alms_A[2], sim_alms_A[3],
                                data_split_alms_B[0], data_split_alms_B[1], data_split_alms_B[2], data_split_alms_B[3])
            #replaces qbXsX
            qCsD = split_K_func(qfunc_CD,
                                sim_alms_C[0], sim_alms_C[1], sim_alms_C[2], sim_alms_C[3],
                                data_split_alms_D[0], data_split_alms_D[1], data_split_alms_D[2], data_split_alms_D[3])

            #qaXXs = split_K_func(Xdat,Xdat1,Xdat2,Xdat3,Xs,Xs1,Xs2,Xs3,qfunc1) #for the two split case, one would need two get_kmaps?, or instead returns an array of maps, [i] for split i
            #qbXXs = split_K_func(Xdat,Xdat1,Xdat2,Xdat3,Xs,Xs1,Xs2,Xs3,qfunc2) if qfunc2 is not None else qaXXs 
            #qaXsX = split_K_func(Xs,Xs1,Xs2,Xs3,Xdat,Xdat1,Xdat2,Xdat3,qfunc1) 
            #qbXsX = split_K_func(Xs,Xs1,Xs2,Xs3,Xdat,Xdat1,Xdat2,Xdat3,qfunc2) if qfunc2 is not None else qaXsX
            
            #rdn0_only_term = power(qABs, qCDs)+ power(qABs, qCsD) + power(qAsB, qCDs) \
            #        + power(qAsB, qCsD)
            rdn0_only_term = power(qABs, qCDs) + power(qAsB, qCDs) + power(qAsB, qCsD) + power(qABs, qCsD)
            
        if verbose:
            print("doing SS terms")
        
        #replaces qaXsXsp
        qAsBsp = split_K_func(qfunc_AB,
                              sim_alms_A[0], sim_alms_A[1], sim_alms_A[2], sim_alms_A[3],
                              sim_alms_Bp[0], sim_alms_Bp[1], sim_alms_Bp[2], sim_alms_Bp[3])
        #replaces qbXsXsp
        qCsDsp = split_K_func(qfunc_CD,
                              sim_alms_C[0], sim_alms_C[1], sim_alms_C[2], sim_alms_C[3],
                              sim_alms_Dp[0], sim_alms_Dp[1], sim_alms_Dp[2], sim_alms_Dp[3])
        #replaces qbXspXs
        qCspDs = split_K_func(qfunc_CD,
                              sim_alms_Cp[0], sim_alms_Cp[1], sim_alms_Cp[2], sim_alms_Cp[3],
                              sim_alms_D[0], sim_alms_D[1], sim_alms_D[2], sim_alms_D[3])
        
        #qaXsXsp = split_K_func(Xs,Xs1,Xs2,Xs3,Xsp,Xsp1,Xsp2,Xsp3,qfunc1)
        #qbXsXsp = split_K_func(Xs,Xs1,Xs2,Xs3,Xsp,Xsp1,Xsp2,Xsp3,qfunc2)
        #qbXspXs = split_K_func(Xsp,Xsp1,Xsp2,Xsp3,Xs,Xs1,Xs2,Xs3,qfunc2)
        #mcn0_term = (power(qaXsXsp,qbXsXsp) + power(qaXsXsp,qbXspXs))
        mcn0_term = (power(qAsBsp, qCsDsp) + power(qAsBsp, qCspDs))

        mcn0evals.append(mcn0_term.copy())
        if not(skip_rd):  rdn0evals.append(rdn0_only_term - mcn0_term)

    if verbose:
        print("combining to get rdn0 and mcn0")
    if not(skip_rd):
        avgrdn0 = utils.allgatherv(rdn0evals,comm)
    else:
        avgrdn0 = None
    avgmcn0 = utils.allgatherv(mcn0evals,comm)
    return avgrdn0, avgmcn0
