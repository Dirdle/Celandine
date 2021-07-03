from __future__ import division

import numpy as np
from scipy.special import factorial2 as fact2
from scipy.special import hyp1f1
from scipy.linalg import fractional_matrix_power as mat_pow
from numpy.linalg import multi_dot as dot

class SCF(object):
    """SCF methods and routines for molecule object"""
    def RHF(self,doPrint=True,DIIS=True,direct=False,tol=1e-12):
        """Routine to compute the RHF energy for a closed shell molecule"""
        self.is_converged = False
        self.delta_energy = 1e20
        self.P_RMS        = 1e20
        self.P_old        = np.zeros((self.nbasis,self.nbasis),dtype='complex')
        self.maxiter = 100
        self.direct = direct
        if self.direct:
            self.incFockRst = False # restart; True shuts off incr. fock build
        self.scrTol     = tol # integral screening tolerance
        self.build(self.direct) # build integrals

        self.P = self.P_old
        self.F = self.Core.astype('complex')

        if DIIS:
            self.fockSet = []
            self.errorSet = []

        for step in range(self.maxiter):
            if step > 0:
                self.F_old      = self.F
                energy_old = self.energy
                # need old P for incremental Fock build
                self.buildFock()
                # now update old P
                self.P_old      = self.P

                # note that extrapolated Fock cannot be used with incrm. Fock
                # build. We make a copy to get the new density only.
                if DIIS:
                    F_diis = self.updateDIIS(self.F,self.P)
                    self.FO = np.dot(self.X.T,np.dot(F_diis,self.X))
            if not DIIS or step == 0:
                self.orthoFock()
            E,self.CO   = np.linalg.eigh(self.FO)

            C      = np.dot(self.X,self.CO)
            self.C      = np.dot(self.X,self.CO)
            self.MO     = E
            self.P = np.dot(C[:,:self.nocc],np.conjugate(C[:,:self.nocc]).T)
            self.computeEnergy()

            if step > 0:
                self.delta_energy = self.energy - energy_old
                self.P_RMS        = np.linalg.norm(self.P - self.P_old)
            FPS = np.dot(self.F,np.dot(self.P,self.S))
            SPF = self.adj(FPS)
            error = np.linalg.norm(FPS - SPF)
            if np.abs(self.P_RMS) < tol or step == (self.maxiter - 1):
                if step == (self.maxiter - 1):
                    print("NOT CONVERGED")
                else:
                    self.is_converged = True
                    FPS = np.dot(self.F,np.dot(self.P,self.S))
                    SPF = self.adj(FPS)
                    error = FPS - SPF
                    self.computeDipole()
                    if doPrint:
                        print("E(SCF)    = ", "{0:.12f}".format(self.energy.real)+ \
                              " in "+str(step)+" iterations")
                        print("  Convergence:")
                        print("    FPS-SPF  = ", np.linalg.norm(error))
                        print("    RMS(P)   = ", "{0:.2e}".format(self.P_RMS.real))
                        print("    dE(SCF)  = ", "{0:.2e}".format(self.delta_energy.real))
                        print("  Dipole X = ", "{0:.8f}".format(self.mu[0].real))
                        print("  Dipole Y = ", "{0:.8f}".format(self.mu[1].real))
                        print("  Dipole Z = ", "{0:.8f}".format(self.mu[2].real))
                    break

    def buildFock(self):
        """Routine to build the AO basis Fock matrix"""
        # if self.direct:
        #     if self.incFockRst: # restart incremental fock build?
        #         self.G = formPT(self.P,np.zeros_like(self.P),self.bfs,
        #                         self.nbasis,self.screen,self.scrTol)
        #         self.G = 0.5*(self.G + self.G.T)
        #         self.F = self.Core.astype('complex') + self.G
        #     else:
        #         self.G = formPT(self.P,self.P_old,self.bfs,self.nbasis,
        #                         self.screen,self.scrTol)
        #         self.G = 0.5*(self.G + self.G.T)
        #         self.F = self.F_old + self.G
        #
        # else:
        self.J = np.einsum('pqrs,sr->pq', self.TwoE.astype('complex'),self.P)
        self.K = np.einsum('psqr,sr->pq', self.TwoE.astype('complex'),self.P)
        self.G = 2.*self.J - self.K
        self.F = self.Core.astype('complex') + self.G

    def orthoFock(self):
        """Routine to orthogonalize the AO Fock matrix to orthonormal basis"""
        self.FO = np.dot(self.X.T,np.dot(self.F,self.X))

    def unOrthoFock(self):
        """Routine to unorthogonalize the orthonormal Fock matrix to AO basis"""
        self.F = np.dot(self.U.T,np.dot(self.FO,self.U))

    def orthoDen(self):
        """Routine to orthogonalize the AO Density matrix to orthonormal basis"""
        self.PO = np.dot(self.U,np.dot(self.P,self.U.T))

    def unOrthoDen(self):
        """Routine to unorthogonalize the orthonormal Density matrix to AO basis"""
        self.P = np.dot(self.X,np.dot(self.PO,self.X.T))

    def computeEnergy(self):
        """Routine to compute the SCF energy"""
        self.el_energy = np.einsum('pq,qp',self.Core+self.F,self.P)
        self.energy    = self.el_energy + self.nuc_energy

    def computeDipole(self):
        """Routine to compute the SCF electronic dipole moment"""
        self.el_energy = np.einsum('pq,qp',self.Core+self.F,self.P)
        for i in range(3):
            self.mu[i] = -2*np.trace(np.dot(self.P,self.M[i])) + sum([atom.charge*(atom.origin[i]-self.center_of_charge[i]) for atom in self.atoms])
        # to debye
        self.mu *= 2.541765

    def adj(self,x):
        """Returns Hermitian adjoint of a matrix"""
        assert x.shape[0] == x.shape[1]
        return np.conjugate(x).T

    def comm(self,A,B):
        """Returns commutator [A,B]"""
        return np.dot(A,B) - np.dot(B,A)

    def updateFock(self):
        """Rebuilds/updates the Fock matrix if you add external fields, etc."""
        self.unOrthoDen()
        self.buildFock()
        self.orthoFock()

    def updateDIIS(self,F,P):
        FPS =   dot([F,P,self.S])
        SPF =   self.adj(FPS)
        # error must be in orthonormal basis
        error = dot([self.X,FPS-SPF,self.X])
        self.fockSet.append(self.F)
        self.errorSet.append(error)
        numFock = len(self.fockSet)
        # limit subspace, hardcoded for now
        if numFock > 8:
            del self.fockSet[0]
            del self.errorSet[0]
            numFock -= 1
        B = np.zeros((numFock + 1,numFock + 1))
        B[-1,:] = B[:,-1] = -1.0
        B[-1,-1] = 0.0
        # B is symmetric
        for i in range(numFock):
            for j in range(i+1):
                B[i,j] = B[j,i] = \
                    np.real(np.trace(np.dot(self.adj(self.errorSet[i]),
                                                     self.errorSet[j])))
        residual = np.zeros(numFock + 1)
        residual[-1] = -1.0
        weights = np.linalg.solve(B,residual)

        # weights is 1 x numFock + 1, but first numFock values
        # should sum to one if we are doing DIIS correctly
        assert np.isclose(sum(weights[:-1]),1.0)

        F = np.zeros((self.nbasis,self.nbasis),dtype='complex')
        for i, Fock in enumerate(self.fockSet):
            F += weights[i] * Fock

        return F


class Forces(object):
    """Nuclear gradient methods and routines for molecule object"""
    def forces(self):
        """Compute the nuclear forces"""

        if not self.is_converged:
            self.exit('Need to converge SCF before computing gradient')

        # get the 3N forces on the molecule
        for atom in self.atoms:
            # reset forces to zero
            atom.forces = np.zeros(3)
            for direction in range(3):
                # init derivative arrays
                dSx = np.zeros_like(self.S)
                dTx = np.zeros_like(self.T)
                dVx = np.zeros_like(self.V)
                dTwoEx = np.zeros_like(self.TwoE)
                dVNx = 0.0
                # do one electron nuclear derivatives
                for i in (range(self.nbasis)):
                    for j in range(i+1):
                        # dSij/dx =
                        #   < d phi_i/ dx | phi_j > + < phi_i | d phi_j / dx >
                        # atom.mask is 1 if the AO involves the nuclei being
                        # differentiated, is 0 if not.
                        dSx[i,j] = dSx[j,i] \
                             = atom.mask[i]*Sx(self.bfs[i],self.bfs[j],
                                               x=direction,center='A') \
                             + atom.mask[j]*Sx(self.bfs[i],self.bfs[j],
                                               x=direction,center='B')
                        # dTij/dx is same form as differentiated overlaps,
                        # since Del^2 does not depend on nuclear origin
                        dTx[i,j] = dTx[j,i] \
                             = atom.mask[i]*Tx(self.bfs[i],self.bfs[j],x=direction,center='A') \
                             + atom.mask[j]*Tx(self.bfs[i],self.bfs[j],x=direction,center='B')
                        # Hellman-feynman term: dVij /dx = < phi_i | d (1/r_c) / dx | phi_j >
                        dVx[i,j] = dVx[j,i] = -atom.charge*VxA(self.bfs[i],self.bfs[j],atom.origin,x=direction)
                        # Terms from deriv of overlap, just like dS/dx and dT/dx
                        for atomic_center in self.atoms:
                            dVx[i,j] -= atom.mask[i]*atomic_center.charge*VxB(self.bfs[i],self.bfs[j],atomic_center.origin,x=direction,center='A')
                            dVx[i,j] -= atom.mask[j]*atomic_center.charge*VxB(self.bfs[i],self.bfs[j],atomic_center.origin,x=direction,center='B')
                        dVx[j,i] = dVx[i,j]

                # do nuclear repulsion contibution
                for atomic_center in self.atoms:
                    # put in A != B conditions
                    RAB = np.linalg.norm(atom.origin - atomic_center.origin)
                    XAB = atom.origin[direction] - atomic_center.origin[direction]
                    ZA  = atom.charge
                    ZB  = atomic_center.charge
                    if not np.allclose(RAB,0.0):
                        dVNx += -XAB*ZA*ZB/(RAB*RAB*RAB)

                # now do two electron contributions
                for i in (range(self.nbasis)):
                    for j in range(i+1):
                        ij = (i*(i+1)//2 + j)
                        for k in range(self.nbasis):
                            for l in range(k+1):
                                kl = (k*(k+1)//2 + l)
                                if ij >= kl:
                                   # do the four terms for gradient two electron
                                   val = atom.mask[i]*ERIx(self.bfs[i],self.bfs[j],self.bfs[k],self.bfs[l],x=direction,center='a')
                                   val += atom.mask[j]*ERIx(self.bfs[i],self.bfs[j],self.bfs[k],self.bfs[l],x=direction,center='b')
                                   val += atom.mask[k]*ERIx(self.bfs[i],self.bfs[j],self.bfs[k],self.bfs[l],x=direction,center='c')
                                   val += atom.mask[l]*ERIx(self.bfs[i],self.bfs[j],self.bfs[k],self.bfs[l],x=direction,center='d')
                                   # we have exploited 8-fold permutaitonal symmetry here
                                   dTwoEx[i,j,k,l] = val
                                   dTwoEx[k,l,i,j] = val
                                   dTwoEx[j,i,l,k] = val
                                   dTwoEx[l,k,j,i] = val
                                   dTwoEx[j,i,k,l] = val
                                   dTwoEx[l,k,i,j] = val
                                   dTwoEx[i,j,l,k] = val
                                   dTwoEx[k,l,j,i] = val

                # Fock gradient terms
                Hx = dTx + dVx
                Jx = np.einsum('pqrs,sr->pq', dTwoEx, self.P)
                Kx = np.einsum('psqr,sr->pq', dTwoEx, self.P)
                Gx = 2.*Jx - Kx
                Fx = Hx + Gx
                force = np.einsum('pq,qp',self.P,Fx + Hx)
                # energy-weighted density matrix for overlap derivative
                W = np.dot(self.P,np.dot(self.F,self.P))
                force -= 2*np.einsum('pq,qp',dSx,W)
                # nuclear-nuclear repulsion contribution
                force += dVNx
                # save forces (not mass weighted) and reset geometry
                # strictly speaking we computed dE/dX, but F = -dE/dX
                atom.forces[direction] = np.real(-force)


class Atom(object):
    """Class for an atom"""
    def __init__(self,charge,mass,origin=np.zeros(3)):
        self.charge = charge
        self.origin = origin
        self.mass   = mass
        # contains forces (not mass-weighted)
        self.forces      = np.zeros(3)
        self.saved_forces  = np.zeros(3)
        self.velocities  = np.zeros(3)

class Molecule(SCF,Forces):
    """Class for a molecule object, consisting of Atom objects
       Requres that molecular geometry, charge, and multiplicity be given as
       input on creation.
    """
    def __init__(self,geometry,basis='sto-3g'):
        # geometry is now specified in imput file
        charge, multiplicity, atomlist = self.read_molecule(geometry)
        self.charge = charge
        self.multiplicity = multiplicity
        self.atoms = atomlist
        self.nelec = sum([atom.charge for atom in atomlist]) - charge
        self.nocc  = self.nelec//2
        self.is_built = False

        # Read in basis set data
        import os
        cur_dir = os.path.dirname(__file__)
        basis_path = 'basis/'+str(basis).lower()+'.gbs'
        basis_file = os.path.join(cur_dir, basis_path)
        self.basis_data = self.getBasis(basis_file)
        self.formBasis()

    @property
    def _forces(self):
        # FIXME: assumes forces have been computed!
        F = []
        for atom in range(len(self.atoms)):
            F.append(self.atoms[atom].forces)
        return np.concatenate(F).reshape(-1,3)



    def formBasis(self):
        """Routine to create the basis from the input molecular geometry and
           basis set. On exit, you should have a basis in self.bfs, which is a
           list of BasisFunction objects. This routine also defines the center
           of nuclear charge.
        """
        self.bfs = []
        for atom in self.atoms:
            for momentum,prims in self.basis_data[atom.charge]:
                exps = [e for e,c in prims]
                coefs = [c for e,c in prims]
                for shell in self.momentum2shell(momentum):
                    #self.bfs.append(BasisFunction(np.asarray(atom.origin),\
                    #    np.asarray(shell),np.asarray(exps),np.asarray(coefs)))
                    self.bfs.append(BasisFunction(np.asarray(atom.origin),
                        np.asarray(shell),len(exps),np.asarray(exps),np.asarray(coefs)))
        self.nbasis = len(self.bfs)
        # create masking vector for geometric derivatives
        idx = 0
        for atom in self.atoms:
            atom.mask = np.zeros(self.nbasis)
            for momentum,prims in self.basis_data[atom.charge]:
                for shell in self.momentum2shell(momentum):
                    atom.mask[idx] = 1.0
                    idx += 1

        # note this is center of positive charge (atoms only, no electrons)
        self.center_of_charge =\
            np.asarray([sum([atom.charge*atom.origin[0] for atom in self.atoms]),
                        sum([atom.charge*atom.origin[1] for atom in self.atoms]),
                        sum([atom.charge*atom.origin[2] for atom in self.atoms])])\
                        * (1./sum([atom.charge for atom in self.atoms]))

    def build(self,direct=False):
        """Routine to build necessary integrals"""
        self.one_electron_integrals()
        if direct:
            # populate dict for screening
            self.screen = {}
            for p in range(self.nbasis):
                for q in range(p + 1):
                    pq = p*(p+1)//2 + q
                    self.screen[pq] = ERI(self.bfs[p],self.bfs[q],self.bfs[p],self.bfs[q])
        else:
            self.two_electron_integrals()
        self.is_built = True

    def momentum2shell(self,momentum):
        """Routine to convert angular momentum to Cartesian shell pair in order
           to create the appropriate BasisFunction object (e.g. form px,py,pz)
        """
        shells = {
            'S' : [(0,0,0)],
            'P' : [(1,0,0),(0,1,0),(0,0,1)],
            'D' : [(2,0,0),(1,1,0),(1,0,1),(0,2,0),(0,1,1),(0,0,2)],
            'F' : [(3,0,0),(2,1,0),(2,0,1),(1,2,0),(1,1,1),(1,0,2),
                   (0,3,0),(0,2,1),(0,1,2), (0,0,3)]
            }
        return shells[str(momentum)]

    def sym2num(self,sym):
        """Routine that converts atomic symbol to atomic number"""
        symbol = [
            "X","H","He",
            "Li","Be","B","C","N","O","F","Ne",
            "Na","Mg","Al","Si","P","S","Cl","Ar",
            "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe",
            "Co", "Ni", "Cu", "Zn",
            "Ga", "Ge", "As", "Se", "Br", "Kr",
            "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru",
            "Rh", "Pd", "Ag", "Cd",
            "In", "Sn", "Sb", "Te", "I", "Xe",
            "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm",  "Eu",
            "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
            "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
            "Tl","Pb","Bi","Po","At","Rn"]
        return symbol.index(str(sym))

    def getBasis(self,filename):
        """Routine to read the basis set files (EMSL Gaussian 94 standard)
           The file is first split into atoms, then iterated through (once).
           At the end we get a basis, which is a dictionary of atoms and their
           basis functions: a tuple of angular momentum and the primitives

           Return: {atom: [('angmom',[(exp,coef),...]), ('angmom',[(exp,...}
        """
        basis = {}

        with open(filename, 'r') as basisset:
            data = basisset.read().split('****')

        # Iterate through all atoms in basis set file
        for i in range(1,len(data)):
            atomData = [x.split() for x in data[i].split('\n')[1:-1]]
            for idx,line in enumerate(atomData):
                # Ignore empty lines
                if not line:
                   pass
                # first line gives atom
                elif idx == 0:
                    assert len(line) == 2
                    atom = self.sym2num(line[0])
                    basis[atom] = []
                    # now set up primitives for particular angular momentum
                    newPrim = True
                # Perform the set up once per angular momentum
                elif idx > 0 and newPrim:
                    momentum  = line[0]
                    numPrims  = int(line[1])
                    newPrim   = False
                    count     = 0
                    prims     = []
                    prims2    = [] # need second list for 'SP' case
                else:
                   # Combine primitives with its angular momentum, add to basis
                   if momentum == 'SP':
                       # Many basis sets share exponents for S and P basis
                       # functions so unfortunately we have to account for this.
                       prims.append((float(line[0].replace('D', 'E')),float(line[1].replace('D', 'E'))))
                       prims2.append((float(line[0].replace('D', 'E')),float(line[2].replace('D', 'E'))))
                       count += 1
                       if count == numPrims:
                           basis[atom].append(('S',prims))
                           basis[atom].append(('P',prims2))
                           newPrim = True
                   else:
                       prims.append((float(line[0].replace('D', 'E')),float(line[1].replace('D', 'E'))))
                       count += 1
                       if count == numPrims:
                           basis[atom].append((momentum,prims))
                           newPrim = True

        return basis

    def read_molecule(self,geometry):
        """Routine to read in the charge, multiplicity, and geometry from the
           input script. Coordinates are assumed to be Angstrom.
           Example:

           geometry = '''
                      0 1
                      H  0.0 0.0 1.2
                      H  0.0 0.0 0.0
                      '''
           self.read_molecule(geometry)

        """
        # atomic masses (isotop avg)
        masses = [0.0,1.008,4.003,6.941,9.012,10.812,12.011,14.007,5.999,
                  18.998,20.180,22.990,24.305,26.982,28.086,30.974,32.066,
                  35.453,39.948]
        f = geometry.split('\n')
        # remove any empty lines
        f = filter(None,f)
        # First line is charge and multiplicity
        atomlist = []
        for line_number,line in enumerate(f):
            if line_number == 0:
                assert len(line.split()) == 2
                charge = int(line.split()[0])
                multiplicity = int(line.split()[1])
            else:
                if len(line.split()) == 0: break
                assert len(line.split()) == 4
                sym = self.sym2num(str(line.split()[0]))
                mass = masses[sym]
                # Convert Angstrom to Bohr (au)
                x   = float(line.split()[1])/0.52917721092
                y   = float(line.split()[2])/0.52917721092
                z   = float(line.split()[3])/0.52917721092
                atom = Atom(charge=sym,mass=mass,
                            origin=np.asarray([x,y,z]))
                atomlist.append(atom)

        return charge, multiplicity, atomlist

    def one_electron_integrals(self):
        """Routine to set up and compute one-electron integrals"""
        N = self.nbasis
        # core integrals
        self.S = np.zeros((N,N))
        self.V = np.zeros((N,N))
        self.T = np.zeros((N,N))
        # dipole integrals
        self.M = np.zeros((3,N,N))
        self.mu = np.zeros(3,dtype='complex')

        # angular momentum
        self.L = np.zeros((3,N,N))

        self.nuc_energy = 0.0
        # Get one electron integrals
        #print "One-electron integrals"

        for i in (range(N)):
            for j in range(i+1):
                self.S[i,j] = self.S[j,i] \
                    = S(self.bfs[i],self.bfs[j])
                self.T[i,j] = self.T[j,i] \
                    = T(self.bfs[i],self.bfs[j])
                self.M[0,i,j] = self.M[0,j,i] \
                    = Mu(self.bfs[i],self.bfs[j],self.center_of_charge,'x')
                self.M[1,i,j] = self.M[1,j,i] \
                    = Mu(self.bfs[i],self.bfs[j],self.center_of_charge,'y')
                self.M[2,i,j] = self.M[2,j,i] \
                    = Mu(self.bfs[i],self.bfs[j],self.center_of_charge,'z')
                for atom in self.atoms:
                    self.V[i,j] += -atom.charge*V(self.bfs[i],self.bfs[j],atom.origin)
                self.V[j,i] = self.V[i,j]

                # RxDel is antisymmetric
                self.L[0,i,j] \
                    = RxDel(self.bfs[i],self.bfs[j],self.center_of_charge,'x')
                self.L[1,i,j] \
                    = RxDel(self.bfs[i],self.bfs[j],self.center_of_charge,'y')
                self.L[2,i,j] \
                    = RxDel(self.bfs[i],self.bfs[j],self.center_of_charge,'z')
                self.L[:,j,i] = -1*self.L[:,i,j]

        # Compute nuclear repulsion energy
        for pair in itertools.combinations(self.atoms,2):
            self.nuc_energy += pair[0].charge*pair[1].charge \
                              / np.linalg.norm(pair[0].origin - pair[1].origin)

        # Preparing for SCF
        self.Core       = self.T + self.V
        self.X          = mat_pow(self.S,-0.5)
        self.U          = mat_pow(self.S,0.5)

    def two_electron_integrals(self):
        """Routine to setup and compute two-electron integrals"""
        N = self.nbasis
        self.TwoE = np.zeros((N,N,N,N))
        self.TwoE = doERIs(N,self.TwoE,self.bfs)
        self.TwoE = np.asarray(self.TwoE)


def E(i,j,t,Qx,a,b):
    ''' Recursive definition of Hermite Gaussian coefficients.
        Returns a float.
        a: orbital exponent on Gaussian 'a' (e.g. alpha in the text)
        b: orbital exponent on Gaussian 'b' (e.g. beta in the text)
        i,j: orbital angular momentum number on Gaussian 'a' and 'b'
        t: number nodes in Hermite (depends on type of integral,
           e.g. always zero for overlap integrals)
        Qx: distance between origins of Gaussian 'a' and 'b'
    '''
    p = a + b
    q = a*b/p
    if (t < 0) or (t > (i + j)):
        # out of bounds for t
        return 0.0
    elif i == j == t == 0:
        # base case
        return np.exp(-q*Qx*Qx) # K_AB
    elif j == 0:
        # decrement index i
        return (1/(2*p))*E(i-1,j,t-1,Qx,a,b) - \
               (q*Qx/a)*E(i-1,j,t,Qx,a,b)    + \
               (t+1)*E(i-1,j,t+1,Qx,a,b)
    else:
        # decrement index j
        return (1/(2*p))*E(i,j-1,t-1,Qx,a,b) + \
               (q*Qx/b)*E(i,j-1,t,Qx,a,b)    + \
               (t+1)*E(i,j-1,t+1,Qx,a,b)

def overlap(a,lmn1,A,b,lmn2,B):
    ''' Evaluates overlap integral between two Gaussians
        Returns a float.
        a:    orbital exponent on Gaussian 'a' (e.g. alpha in the text)
        b:    orbital exponent on Gaussian 'b' (e.g. beta in the text)
        lmn1: int tuple containing orbital angular momentum (e.g. (1,0,0))
              for Gaussian 'a'
        lmn2: int tuple containing orbital angular momentum for Gaussian 'b'
        A:    list containing origin of Gaussian 'a', e.g. [1.0, 2.0, 0.0]
        B:    list containing origin of Gaussian 'b'
    '''
    l1,m1,n1 = lmn1 # shell angular momentum on Gaussian 'a'
    l2,m2,n2 = lmn2 # shell angular momentum on Gaussian 'b'
    S1 = E(l1,l2,0,A[0]-B[0],a,b) # X
    S2 = E(m1,m2,0,A[1]-B[1],a,b) # Y
    S3 = E(n1,n2,0,A[2]-B[2],a,b) # Z
    return S1*S2*S3*np.power(np.pi/(a+b),1.5)

def S(a,b):
    '''Evaluates overlap between two contracted Gaussians
       Returns float.
       Arguments:
       a: contracted Gaussian 'a', BasisFunction object
       b: contracted Gaussian 'b', BasisFunction object
    '''
    s = 0.0
    for ia, ca in enumerate(a.coefs):
        for ib, cb in enumerate(b.coefs):
            s += a.norm[ia]*b.norm[ib]*ca*cb*\
                     overlap(a.exps[ia],a.shell,a.origin,
                     b.exps[ib],b.shell,b.origin)
    return s

class BasisFunction(object):
    ''' A class that contains all our basis function data
        Attributes:
        origin: array/list containing the coordinates of the Gaussian origin
        shell:  tuple of angular momentum
        exps:   list of primitive Gaussian exponents
        coefs:  list of primitive Gaussian coefficients
        norm:   list of normalization factors for Gaussian primitives
    '''
    def __init__(self,origin=[0.0,0.0,0.0],shell=(0,0,0),num_exps=None,exps=[],coefs=[]):
        self.origin = np.asarray(origin)
        self.shell = shell
        self.exps  = exps
        self.coefs = coefs
        self.num_exps = len(self.exps)
        self.norm = None
        self.normalize()

    def __repr__(self):
        return "BasisFunction:" + str([x for t in zip(self.exps, self.coefs) for x in t])

    def __str__(self):
        return "BasisFunction at " + str(self.origin) + \
        "\n exponents: " + str(self.exps) + \
        "\n coefficients: " + str(self.coefs) + \
        "\n normalisation: " + str(self.norm)

    def normalize(self):
        ''' Routine to normalize the basis functions, in case they
            do not integrate to unity.
        '''
        l,m,n = self.shell
        L = l+m+n
        # self.norm is a list of length equal to number primitives
        # normalize primitives first (PGBFs)
        self.norm = np.sqrt(np.power(2,2*(l+m+n)+1.5)*
                        np.power(self.exps,l+m+n+1.5)/
                        fact2(2*l-1)/fact2(2*m-1)/
                        fact2(2*n-1)/np.power(np.pi,1.5))

        # now normalize the contracted basis functions (CGBFs)
        # Eq. 1.44 of Valeev integral whitepaper
        prefactor = np.power(np.pi,1.5)*\
            fact2(2*l - 1)*fact2(2*m - 1)*fact2(2*n - 1)/np.power(2.0,L)

        N = 0.0
        num_exps = len(self.exps)
        for ia in range(num_exps):
            for ib in range(num_exps):
                N += self.norm[ia]*self.norm[ib]*self.coefs[ia]*self.coefs[ib]/\
                         np.power(self.exps[ia] + self.exps[ib],L+1.5)

        N *= prefactor
        N = np.power(N,-0.5)
        for ia in range(num_exps):
            self.coefs[ia] *= N

def kinetic(a,lmn1,A,b,lmn2,B):
    ''' Evaluates kinetic energy integral between two Gaussians
        Returns a float.
        a:    orbital exponent on Gaussian 'a' (e.g. alpha in the text)
        b:    orbital exponent on Gaussian 'b' (e.g. beta in the text)
        lmn1: int tuple containing orbital angular momentum (e.g. (1,0,0))
              for Gaussian 'a'
        lmn2: int tuple containing orbital angular momentum for Gaussian 'b'
        A:    list containing origin of Gaussian 'a', e.g. [1.0, 2.0, 0.0]
        B:    list containing origin of Gaussian 'b'
    '''
    l1,m1,n1 = lmn1
    l2,m2,n2 = lmn2
    term0 = b*(2*(l2+m2+n2)+3)*\
                            overlap(a,(l1,m1,n1),A,b,(l2,m2,n2),B)
    term1 = -2*np.power(b,2)*\
                           (overlap(a,(l1,m1,n1),A,b,(l2+2,m2,n2),B) +
                            overlap(a,(l1,m1,n1),A,b,(l2,m2+2,n2),B) +
                            overlap(a,(l1,m1,n1),A,b,(l2,m2,n2+2),B))
    term2 = -0.5*(l2*(l2-1)*overlap(a,(l1,m1,n1),A,b,(l2-2,m2,n2),B) +
                  m2*(m2-1)*overlap(a,(l1,m1,n1),A,b,(l2,m2-2,n2),B) +
                  n2*(n2-1)*overlap(a,(l1,m1,n1),A,b,(l2,m2,n2-2),B))
    return term0+term1+term2

def T(a,b):
    '''Evaluates kinetic energy between two contracted Gaussians
       Returns float.
       Arguments:
       a: contracted Gaussian 'a', BasisFunction object
       b: contracted Gaussian 'b', BasisFunction object
    '''
    t = 0.0
    for ia, ca in enumerate(a.coefs):
        for ib, cb in enumerate(b.coefs):
            t += a.norm[ia]*b.norm[ib]*ca*cb*\
                     kinetic(a.exps[ia],a.shell,a.origin,\
                     b.exps[ib],b.shell,b.origin)
    return t

def R(t,u,v,n,p,PCx,PCy,PCz,RPC):
    ''' Returns the Coulomb auxiliary Hermite integrals
        Returns a float.
        Arguments:
        t,u,v:   order of Coulomb Hermite derivative in x,y,z
                 (see defs in Helgaker and Taylor)
        n:       order of Boys function
        PCx,y,z: Cartesian vector distance between Gaussian
                 composite center P and nuclear center C
        RPC:     Distance between P and C
    '''
    T = p*RPC*RPC
    val = 0.0
    if t == u == v == 0:
        print("Boys " + str(n) + " " + str(T))
        val += np.power(-2*p,n)*boys(n,T)
        print(val)
    elif t == u == 0:
        if v > 1:
            val += (v-1)*R(t,u,v-2,n+1,p,PCx,PCy,PCz,RPC)
        val += PCz*R(t,u,v-1,n+1,p,PCx,PCy,PCz,RPC)
    elif t == 0:
        if u > 1:
            val += (u-1)*R(t,u-2,v,n+1,p,PCx,PCy,PCz,RPC)
        val += PCy*R(t,u-1,v,n+1,p,PCx,PCy,PCz,RPC)
    else:
        if t > 1:
            val += (t-1)*R(t-2,u,v,n+1,p,PCx,PCy,PCz,RPC)
        val += PCx*R(t-1,u,v,n+1,p,PCx,PCy,PCz,RPC)
    return val

def boys(n,T):
    return hyp1f1(n+0.5,n+1.5,-T)/(2.0*n+1.0)

def gaussian_product_center(a,A,b,B):
    return (a*A+b*B)/(a+b)

def nuclear_attraction(a,lmn1,A,b,lmn2,B,C):
    ''' Evaluates kinetic energy integral between two Gaussians
         Returns a float.
         a:    orbital exponent on Gaussian 'a' (e.g. alpha in the text)
         b:    orbital exponent on Gaussian 'b' (e.g. beta in the text)
         lmn1: int tuple containing orbital angular momentum (e.g. (1,0,0))
               for Gaussian 'a'
         lmn2: int tuple containing orbital angular momentum for Gaussian 'b'
         A:    list containing origin of Gaussian 'a', e.g. [1.0, 2.0, 0.0]
         B:    list containing origin of Gaussian 'b'
         C:    list containing origin of nuclear center 'C'
    '''
    l1,m1,n1 = lmn1
    l2,m2,n2 = lmn2
    p = a + b
    P = gaussian_product_center(a,A,b,B) # Gaussian composite center
    RPC = np.linalg.norm(P-C)

    val = 0.0
    for t in range(l1+l2+1):
        for u in range(m1+m2+1):
            for v in range(n1+n2+1):
                val += E(l1,l2,t,A[0]-B[0],a,b) * \
                       E(m1,m2,u,A[1]-B[1],a,b) * \
                       E(n1,n2,v,A[2]-B[2],a,b) * \
                       R(t,u,v,0,p,P[0]-C[0],P[1]-C[1],P[2]-C[2],RPC)
    val *= 2*np.pi/p
    return val

def V(a,b,C):
    '''Evaluates overlap between two contracted Gaussians
       Returns float.
       Arguments:
       a: contracted Gaussian 'a', BasisFunction object
       b: contracted Gaussian 'b', BasisFunction object
       C: center of nucleus
    '''
    v = 0.0
    for ia, ca in enumerate(a.coefs):
        for ib, cb in enumerate(b.coefs):
            v += a.norm[ia]*b.norm[ib]*ca*cb*\
                     nuclear_attraction(a.exps[ia],a.shell,a.origin,
                     b.exps[ib],b.shell,b.origin,C)
    return v

def electron_repulsion(a,lmn1,A,b,lmn2,B,c,lmn3,C,d,lmn4,D):
    ''' Evaluates kinetic energy integral between two Gaussians
        Returns a float.
        a,b,c,d:   orbital exponent on Gaussian 'a','b','c','d'
        lmn1,lmn2
        lmn3,lmn4: int tuple containing orbital angular momentum
                   for Gaussian 'a','b','c','d', respectively
        A,B,C,D:   list containing origin of Gaussian 'a','b','c','d'
    '''
    l1,m1,n1 = lmn1
    l2,m2,n2 = lmn2
    l3,m3,n3 = lmn3
    l4,m4,n4 = lmn4
    p = a+b # composite exponent for P (from Gaussians 'a' and 'b')
    q = c+d # composite exponent for Q (from Gaussians 'c' and 'd')
    alpha = p*q/(p+q)
    P = gaussian_product_center(a,A,b,B) # A and B composite center
    Q = gaussian_product_center(c,C,d,D) # C and D composite center
    RPQ = np.linalg.norm(P-Q)

    val = 0.0
    for t in range(l1+l2+1):
        for u in range(m1+m2+1):
            for v in range(n1+n2+1):
                for tau in range(l3+l4+1):
                    for nu in range(m3+m4+1):
                        for phi in range(n3+n4+1):
                            val += E(l1,l2,t,A[0]-B[0],a,b) * \
                                   E(m1,m2,u,A[1]-B[1],a,b) * \
                                   E(n1,n2,v,A[2]-B[2],a,b) * \
                                   E(l3,l4,tau,C[0]-D[0],c,d) * \
                                   E(m3,m4,nu ,C[1]-D[1],c,d) * \
                                   E(n3,n4,phi,C[2]-D[2],c,d) * \
                                   np.power(-1,tau+nu+phi) * \
                                   R(t+tau,u+nu,v+phi,0,\
                                       alpha,P[0]-Q[0],P[1]-Q[1],P[2]-Q[2],RPQ)

    val *= 2*np.power(np.pi,2.5)/(p*q*np.sqrt(p+q))
    return val

def ERI(a,b,c,d):
    '''Evaluates overlap between two contracted Gaussians
        Returns float.
        Arguments:
        a: contracted Gaussian 'a', BasisFunction object
        b: contracted Gaussian 'b', BasisFunction object
        c: contracted Gaussian 'b', BasisFunction object
        d: contracted Gaussian 'b', BasisFunction object
    '''
    eri = 0.0
    for ja, ca in enumerate(a.coefs):
        for jb, cb in enumerate(b.coefs):
            for jc, cc in enumerate(c.coefs):
                for jd, cd in enumerate(d.coefs):
                    eri += a.norm[ja]*b.norm[jb]*c.norm[jc]*d.norm[jd]*\
                             ca*cb*cc*cd*\
                             electron_repulsion(a.exps[ja],a.shell,a.origin,\
                                                b.exps[jb],b.shell,b.origin,\
                                                c.exps[jc],c.shell,c.origin,\
                                                d.exps[jd],d.shell,d.origin)
    return eri


if __name__ == "__main__":
    # H_one = BasisFunction(origin = [0.866812, 0.601436, 0.0], exps = [3.42525, 0.623914, 0.168855], coefs = [0.154329, 0.535328, 0.444635])
    # print(H_one)
    # H_two = BasisFunction(origin = [-0.866812, 0.601436, 0.0], exps = [3.42525, 0.623914, 0.168855], coefs = [0.154329, 0.535328, 0.444635])
    # print(S(H_one, H_two))

    #Create new water molecule
    waterInp = """
    0 1
    O    0.000000      -0.075791844    0.000000
    H    0.866811829    0.601435779    0.000000
    H   -0.866811829    0.601435779    0.000000
    """

    water = Molecule(waterInp, "sto-3g")
    for bf in water.bfs:
        print(bf)
