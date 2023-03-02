import pixell.curvedsky as cs
import numpy as np

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

def mcrdn0_s4(icov, get_kmap, power, nsims, qfunc1,get_kmap1=None,get_kmap2=None,get_kmap3=None, qfunc2=None, Xdat=None,Xdat1=None,Xdat2=None,Xdat3=None, use_mpi=True, 
              verbose=True, skip_rd=False,shear=False,power_mcn0=None, phifunc=None):
         
    qa = phifunc
    if qa is None:
        def qa(X):
            return X
    qf1 = qfunc1
    qf2 = qfunc2
    
    mcn0evals = []
    if not(skip_rd): 
        assert Xdat is not None # Data
        if Xdat1 is None:
            Xdat1=Xdat
        rdn0evals = []

    if use_mpi:
        comm,rank,my_tasks = mpi.distribute(nsims)
    else:
        comm,rank,my_tasks = FakeCommunicator(), 0, range(nsims)
        

    for i in my_tasks:
        i=i+2
        if rank==0 and verbose: print("MCRDN0: Rank %d doing task %d" % (rank,i))
        Xs  = get_kmap((icov,0,i))
        Xs1= get_kmap1((icov,0,i))
        Xs2= get_kmap2((icov,0,i))
        Xs3= get_kmap3((icov,0,i))


        if not(skip_rd): 
            qaXXs = qa(Xdat,Xdat1,Xdat2,Xdat3,Xs,Xs1,Xs2,Xs3,qf1) #for the two split case, one would need two get_kmaps?, or instead returns an array of maps, [i] for split i
            qbXXs = qa(Xdat,Xdat1,Xdat2,Xdat3,Xs,Xs1,Xs2,Xs3,qf2) if qf2 is not None else qaXXs 
            qaXsX = qa(Xs,Xs1,Xs2,Xs3,Xdat,Xdat1,Xdat2,Xdat3,qf1) 
            qbXsX = qa(Xs,Xs1,Xs2,Xs3,Xdat,Xdat1,Xdat2,Xdat3,qf2) if qf2 is not None else qaXsX 
            rdn0_only_term = power(qaXXs,qbXXs)+ power(qaXXs,qbXsX) + power(qaXsX,qbXXs) \
                    + power(qaXsX,qbXsX) 

        Xsp = get_kmap((icov,0,i+1)) 
        Xsp1 = get_kmap1((icov,0,i+1)) 
        Xsp2 = get_kmap2((icov,0,i+1)) 
        Xsp3 = get_kmap3((icov,0,i+1)) 

        if shear:
            qaXsXsp = plensing.phi_to_kappa(qf1(Xs[0],Xsp[1])) #split1 
            qbXsXsp = plensing.phi_to_kappa(qf2(Xs[0],Xsp[1])) if qf2 is not None else qaXsXsp #split2
            qbXspXs = plensing.phi_to_kappa(qf2(Xsp[0],Xs[1])) if qf2 is not None else plensing.phi_to_kappa(qf1(Xsp[0],Xs[1])) #this is not present
        else:
            if power_mcn0 is None:
                qaXsXsp = qa(Xs,Xs1,Xs2,Xs3,Xsp,Xsp1,Xsp2,Xsp3,qf1) #split1 
                qbXsXsp = qa(Xs,Xs1,Xs2,Xs3,Xsp,Xsp1,Xsp2,Xsp3,qf2) if qf2 is not None else qaXsXsp #split2
                qbXspXs = qa(Xsp,Xsp1,Xsp2,Xsp3,Xs,Xs1,Xs2,Xs3,qf2) if qf2 is not None else qa(Xsp,Xsp1,Xsp2,Xsp3,Xs,Xs1,Xs2,Xs3,qf1) #this is not present
                mcn0_term = (power(qaXsXsp,qbXsXsp) + power(qaXsXsp,qbXspXs))
            else:
                qaXsXsp = plensing.phi_to_kappa(qf1(Xs,Xsp)) #split1 
                qbXsXsp = plensing.phi_to_kappa(qf2(Xs,Xsp)) if qf2 is not None else qaXsXsp #split2
                qbXspXs = plensing.phi_to_kappa(qf2(Xsp,Xs)) if qf2 is not None else plensing.phi_to_kappa(qf1(Xsp,Xs)) #this is not present
                mcn0_term = (power_mcn0(qaXsXsp,qbXsXsp) + power_mcn0(qaXsXsp,qbXspXs))

        mcn0evals.append(mcn0_term.copy())
        if not(skip_rd):  rdn0evals.append(rdn0_only_term - mcn0_term)

    if not(skip_rd):
        avgrdn0 = utils.allgatherv(rdn0evals,comm)
    else:
        avgrdn0 = None
    avgmcn0 = utils.allgatherv(mcn0evals,comm)
    return avgrdn0, avgmcn0
