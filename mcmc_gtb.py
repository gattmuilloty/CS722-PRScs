#!/usr/bin/env python

from scipy import linalg
import numpy as np
import parseAndGigrnd


def updateBeta(beta, LDBlock, blockSizes, numBlocks, marginalBeta, sigma, n, psi):
    # Initialize values
    mm = 0; quad = 0.0
    for kk in range(numBlocks):
        if blockSizes[kk] == 0:
            continue
        else:
            # Get indexes of the LD block
            blockRange = range(mm,mm+blockSizes[kk])
            # T-inverse ( diag{Phi * Psi} )
            T_inv = np.diag(1.0/psi[blockRange].T[0])
            # Updated effect sizes
            # tempBeta = linalg.solve(linalg.cholesky(LDBlock[kk] + T_inv).T, marginalBeta[blockRange]) + np.sqrt(sigma/n)*np.random.randn(len(blockRange),1)
            beta[blockRange] = linalg.solve(linalg.cholesky(LDBlock[kk] + T_inv), linalg.solve(linalg.cholesky(LDBlock[kk] + T_inv).T, marginalBeta[blockRange]) + np.sqrt(sigma/n)*np.random.randn(len(blockRange),1))
            # Quadratic term
            quad += np.dot(np.dot(beta[blockRange].T, LDBlock[kk]+T_inv), beta[blockRange])
            mm += blockSizes[kk]
    return beta, quad


def mcmc(a, b, phi, summaryStats, n, LDBlock, blockSizes, n_iter, n_burnin, thin, chrom, out_dir, seed):
    print('... MCMC ...')

    # Initialize random seed
    if seed != None:
        np.random.seed(seed)
        
    # Estimated effect sizes
    marginalBeta = np.array(summaryStats['BETA'], ndmin=2).T

    # Number of posterior statistics
    numPosteriorStats = int((n_iter-n_burnin)/thin)

    # Number of SNPs (from GWAS Summary Statistics)
    numSNPs = len(summaryStats['SNP'])

    # Number of LD blocks (from LD Reference Panel) 
    numBlocks = len(LDBlock)

    # Initialize effect sizes as an array of zeros
    beta = np.zeros((numSNPs,1))

    # Initialize the local shrinkage parameters to an array of ones
    psi = np.ones((numSNPs,1))
    
    # Unit variance/standard deviation
    sigma = 1.0

    # Initialize estimated parameters
    estimatedBeta = np.zeros((numSNPs,1))
    estimatedPsi = np.zeros((numSNPs,1))
    estimatedSigma = 0.0
    estimatedPhi = 0.0

    # MCMC
    pp = 0
    for itr in range(1,n_iter+1):

        # For each 100th iteration
        if itr % 100 == 0:
            print(' - Interation #' + str(itr) + ' -')

        #### Update Beta (Effect Sizes)
        beta, quad = updateBeta(beta, LDBlock, blockSizes, numBlocks, marginalBeta, sigma, n, psi)

        # Error
        err = max(n/2.0*(1.0-2.0*sum(beta*marginalBeta)+quad), n/2.0*sum(beta**2/psi))

        sigma = 1.0/np.random.gamma((n+numSNPs)/2.0, 1.0/err)

        # Update the gamma-gamma parameters
        delta = np.random.gamma(a+b, 1.0/(psi+phi))

        for jj in range(numSNPs):
            psi[jj] = parseAndGigrnd.gigrnd(a-0.5, 2.0*delta[jj], n*beta[jj]**2/sigma)

        psi[psi>1] = 1.0

        # Update posterior stats
        if (itr>n_burnin) and (itr % thin == 0):
            estimatedBeta = estimatedBeta + beta/numPosteriorStats
            estimatedPsi = estimatedPsi + psi/numPosteriorStats
            estimatedSigma = estimatedSigma + sigma/numPosteriorStats
            estimatedPhi = estimatedPhi + phi/numPosteriorStats
            
    #
    #
    #
    # Write posterior effect sizes
    #
    #
    #
    eff_file = out_dir + '_pst_eff_a%d_b%.1f_phi%1.0e_chr%d.txt' % (a, b, phi, chrom)

    columns = ['SNP', 'BP', 'A1', 'A2', 'BETA']
    summaryStats['BETA'] = estimatedBeta
    summaryStats[columns].to_csv(eff_file, sep='\t', index=False, header=False)

    print('... Done ...')