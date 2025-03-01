# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
Coupled Cluster
===============

Simple usage::

    >>> from pyscf import gto, scf, cc
    >>> mol = gto.M(atom='H 0 0 0; H 0 0 1')
    >>> mf = scf.RHF(mol).run()
    >>> cc.CCSD(mf).run()

:func:`cc.CCSD` returns an instance of CCSD class.  Following are parameters
to control CCSD calculation.

    verbose : int
        Print level.  Default value equals to :class:`Mole.verbose`
    max_memory : float or int
        Allowed memory in MB.  Default value equals to :class:`Mole.max_memory`
    conv_tol : float
        converge threshold.  Default is 1e-7.
    conv_tol_normt : float
        converge threshold for norm(t1,t2).  Default is 1e-5.
    max_cycle : int
        max number of iterations.  Default is 50.
    diis_space : int
        DIIS space size.  Default is 6.
    diis_start_cycle : int
        The step to start DIIS.  Default is 0.
    direct : bool
        AO-direct CCSD. Default is False.
    async_io : bool
        Allow for asynchronous function execution. Default is True.
    incore_complete : bool
        Avoid all I/O. Default is False.
    frozen : int or list
        If integer is given, the inner-most orbitals are frozen from CC
        amplitudes.  Given the orbital indices (0-based) in a list, both
        occupied and virtual orbitals can be frozen in CC calculation.


Saved results

    iterinfo : common.IterationInfo
        Information about iteration (see pyscf.common.Iteration in detail)
    e_tot : float
        Total CCSD energy (HF + correlation)
    t1, t2 :
        t1[i,a], t2[i,j,a,b]  (i,j in occ, a,b in virt)
    l1, l2 :
        Lambda amplitudes l1[i,a], l2[i,j,a,b]  (i,j in occ, a,b in virt)
'''

from pyscf.cc import ccsd
from pyscf.cc import ccsd_lambda
from pyscf.cc import ccsd_rdm
from pyscf.cc import addons
from pyscf.cc import rccsd
from pyscf.cc import uccsd
from pyscf.cc import gccsd
from pyscf.cc import eom_rccsd
from pyscf.cc import eom_uccsd
from pyscf.cc import eom_gccsd
from pyscf.cc import qcisd
from pyscf import scf

def CCSD(mf, frozen=None, mo_coeff=None, mo_occ=None):
    if mf.istype('UHF'):
        return UCCSD(mf, frozen, mo_coeff, mo_occ)
    elif mf.istype('GHF'):
        return GCCSD(mf, frozen, mo_coeff, mo_occ)
    else:
        return RCCSD(mf, frozen, mo_coeff, mo_occ)
CCSD.__doc__ = ccsd.CCSD.__doc__

scf.hf.SCF.CCSD = CCSD


def RCCSD(mf, frozen=None, mo_coeff=None, mo_occ=None):
    import numpy
    from pyscf import lib
    from pyscf.df.df_jk import _DFHF
    from pyscf.cc import dfccsd

    if mf.istype('UHF'):
        raise RuntimeError('RCCSD cannot be used with UHF method.')
    elif mf.istype('ROHF'):
        lib.logger.warn(mf, 'RCCSD method does not support ROHF method. ROHF object '
                        'is converted to UHF object and UCCSD method is called.')
        mf = mf.to_uhf()
        return UCCSD(mf, frozen, mo_coeff, mo_occ)

    mf = mf.remove_soscf()
    if not mf.istype('RHF'):
        mf = mf.to_rhf()

    if isinstance(mf, _DFHF) and mf.with_df:
        return dfccsd.RCCSD(mf, frozen, mo_coeff, mo_occ)

    elif numpy.iscomplexobj(mo_coeff) or numpy.iscomplexobj(mf.mo_coeff):
        return rccsd.RCCSD(mf, frozen, mo_coeff, mo_occ)

    else:
        return ccsd.CCSD(mf, frozen, mo_coeff, mo_occ)
RCCSD.__doc__ = ccsd.CCSD.__doc__


def UCCSD(mf, frozen=None, mo_coeff=None, mo_occ=None):
    from pyscf.df.df_jk import _DFHF

    mf = mf.remove_soscf()
    if not mf.istype('UHF'):
        mf = mf.to_uhf()

    if isinstance(mf, _DFHF) and mf.with_df:
        # TODO: DF-UCCSD with memory-efficient particle-particle ladder,
        # similar to dfccsd.RCCSD
        return uccsd.UCCSD(mf, frozen, mo_coeff, mo_occ)
    else:
        return uccsd.UCCSD(mf, frozen, mo_coeff, mo_occ)
UCCSD.__doc__ = uccsd.UCCSD.__doc__


def GCCSD(mf, frozen=None, mo_coeff=None, mo_occ=None):
    from pyscf.df.df_jk import _DFHF

    mf = mf.remove_soscf()
    if not mf.istype('GHF'):
        mf = mf.to_ghf()

    if isinstance(mf, _DFHF) and mf.with_df:
        raise NotImplementedError('DF-GCCSD')
    else:
        return gccsd.GCCSD(mf, frozen, mo_coeff, mo_occ)
GCCSD.__doc__ = gccsd.GCCSD.__doc__


def QCISD(mf, frozen=None, mo_coeff=None, mo_occ=None):
    if mf.istype('UHF'):
        raise NotImplementedError
    elif mf.istype('GHF'):
        raise NotImplementedError
    else:
        return RQCISD(mf, frozen, mo_coeff, mo_occ)
QCISD.__doc__ = qcisd.QCISD.__doc__

scf.hf.SCF.QCISD = QCISD

def RQCISD(mf, frozen=None, mo_coeff=None, mo_occ=None):
    import numpy
    from pyscf import lib

    if mf.istype('UHF'):
        raise RuntimeError('RQCISD cannot be used with UHF method.')
    elif mf.istype('ROHF'):
        lib.logger.warn(mf, 'RQCISD method does not support ROHF method. ROHF object '
                        'is converted to UHF object and UQCISD method is called.')
        raise NotImplementedError

    mf = mf.remove_soscf()
    if not mf.istype('RHF'):
        mf = mf.to_rhf()

    elif numpy.iscomplexobj(mo_coeff) or numpy.iscomplexobj(mf.mo_coeff):
        raise NotImplementedError

    else:
        return qcisd.QCISD(mf, frozen, mo_coeff, mo_occ)
RQCISD.__doc__ = qcisd.QCISD.__doc__


def FNOCCSD(mf, thresh=1e-6, pct_occ=None, nvir_act=None, frozen=None):
    """Frozen natural orbital CCSD

    Attributes:
        thresh : float
            Threshold on NO occupation numbers. Default is 1e-6 (very conservative).
        pct_occ : float
            Percentage of total occupation number. Default is None. If present, overrides `thresh`.
        nvir_act : int
            Number of virtual NOs to keep. Default is None. If present, overrides `thresh` and `pct_occ`.
    """
    import numpy
    from pyscf import mp
    pt = mp.MP2(mf, frozen=frozen).set(verbose=3).run()
    frozen, no_coeff = pt.make_fno(thresh=thresh, pct_occ=pct_occ, nvir_act=nvir_act)
    if len(frozen) == 0: frozen = None
    pt_no = mp.MP2(mf, frozen=frozen, mo_coeff=no_coeff).set(verbose=3).run()
    mycc = CCSD(mf, frozen=frozen, mo_coeff=no_coeff)
    mycc.delta_emp2 = pt.e_corr - pt_no.e_corr
    print('FNOCCSD: Delta E(MP2) =', mycc.delta_emp2)
    # frozen_flat = numpy.asarray(frozen)  # 将所有数组展平为一个长数组
    # mycc.total_count = len(frozen_flat)
    from pyscf.lib import logger
    def _finalize(self):
        '''Hook for dumping results and clearing up the object.'''
        if self.converged:
            logger.info(self, 'FNO-%s converged', self.__class__.__name__)
        else:
            logger.note(self, 'FNO-%s not converged', self.__class__.__name__)
        logger.note(self, 'E(FNO-%s) = %.16g  E_corr = %.16g',
                    self.__class__.__name__, self.e_tot, self.e_corr)
        logger.note(self, 'E(FNO-%s+delta-MP2) = %.16g  E_corr = %.16g',
                    self.__class__.__name__, self.e_tot+self.delta_emp2,
                    self.e_corr+self.delta_emp2)
        return self
    mycc._finalize = _finalize.__get__(mycc, mycc.__class__)
    return mycc

def FNOCCSD_rpa(mf, thresh=1e-6, pct_occ=None, nvir_act=None, frozen=None):
    """Frozen natural orbital CCSD

    Attributes:
        thresh : float
            Threshold on NO occupation numbers. Default is 1e-6 (very conservative).
        pct_occ : float
            Percentage of total occupation number. Default is None. If present, overrides `thresh`.
        nvir_act : int
            Number of virtual NOs to keep. Default is None. If present, overrides `thresh` and `pct_occ`.
    """
    import numpy
    from pyscf import mp
    import sys
    sys.path.append('/Users/xlgong/software/lno/lno/rpa')
    import rpa

    pt = rpa.RPA(mf, frozen=frozen).set(verbose=0).run()
    nocc= pt.nocc
    print('shape of pt.mo_coeff is ', pt.mo_coeff[0])
    frozen, no_coeff = mp.mp2.make_fno(pt, thresh=thresh, pct_occ=pct_occ, nvir_act=nvir_act)
    print('shape of no_coeff is ', no_coeff[0])
    if len(frozen) == 0: frozen = None
    pt_no =rpa.RPA(mf, frozen=frozen, mo_coeff=no_coeff).set(verbose=3).run()

    print('shape of pt_no.mo_coeff is ', pt_no.mo_coeff[1])
    mycc = CCSD(mf, frozen=frozen, mo_coeff=no_coeff)
    mycc.delta_erpa = pt.e_corr - pt_no.e_corr
    print('FNOCCSD: Delta E(RPA) =', mycc.delta_erpa)
    print(frozen)
    if frozen is not None:
        mycc.total_count = len(frozen)
    else:
        mycc.total_count = 0  # 或者其他合理的默认值
    print(f"Total number of frozen orbitals: {mycc.total_count}")
    from pyscf.lib import logger
    def _finalize(self):
        '''Hook for dumping results and clearing up the object.'''
        if self.converged:
            logger.info(self, 'FNO-%s converged', self.__class__.__name__)
        else:
            logger.note(self, 'FNO-%s not converged', self.__class__.__name__)
        logger.note(self, 'E(FNO-%s) = %.16g  E_corr = %.16g',
                    self.__class__.__name__, self.e_tot, self.e_corr)
        logger.note(self, 'E(FNO-%s+delta-MP2) = %.16g  E_corr = %.16g',
                    self.__class__.__name__, self.e_tot+self.delta_erpa,
                    self.e_corr+self.delta_erpa)
        return self
    mycc._finalize = _finalize.__get__(mycc, mycc.__class__)
    return mycc

def FNOCCSD_krpa(mf, thresh=1e-6, pct_occ=None, nvir_act=None, frozen=None):
    """Frozen natural orbital CCSD

    Attributes:
        thresh : float
            Threshold on NO occupation numbers. Default is 1e-6 (very conservative).
        pct_occ : float
            Percentage of total occupation number. Default is None. If present, overrides `thresh`.
        nvir_act : int
            Number of virtual NOs to keep. Default is None. If present, overrides `thresh` and `pct_occ`.
    """
    import numpy
    from pyscf.pbc import cc
    import sys
    sys.path.append('/Users/xlgong/software/lno/lno/rpa')
    import krpa


    pt = krpa.KRPA(mf, frozen=frozen).set(verbose=6).run()
    no_energy, frozen, no_coeff= krpa.KRPA.make_fno(pt, thresh=thresh, pct_occ=pct_occ, nvir_act=nvir_act)
    # print(no_coeff[0])
    print(frozen)
    if len(frozen) == 0: frozen = None   

    pt_no =krpa.KRPA(mf, frozen=frozen, mo_coeff=no_coeff, mo_energy = no_energy).set(verbose=5)
    # pt_no.mo_energy = no_energy
    pt_no.kernel()
    mycc = cc.KCCSD(mf, frozen=frozen, mo_coeff=no_coeff)
    mycc.delta_ekrpa = pt.e_corr - pt_no.e_corr
    print('FNOCCSD: Delta E(krpa) =', mycc.delta_ekrpa)
    frozen_flat = numpy.concatenate(frozen)  
    mycc.total_count = len(frozen_flat)
    print(f"Total number of frozen orbitals: {mycc.total_count}")
    mycc.frozen = frozen
    from pyscf.lib import logger
    def _finalize(self):
        '''Hook for dumping results and clearing up the object.'''
        if self.converged:
            logger.info(self, 'FNO-%s converged', self.__class__.__name__)
        else:
            logger.note(self, 'FNO-%s not converged', self.__class__.__name__)
        logger.note(self, 'E(FNO-%s) = %.16g  E_corr = %.16g',
                    self.__class__.__name__, self.e_tot, self.e_corr)
        logger.note(self, 'E(FNO-%s+delta-krpa) = %.16g  E_corr = %.16g',
                    self.__class__.__name__,self.e_tot+self.delta_ekrpa,
                    self.e_corr+self.delta_ekrpa) 
        return self
    mycc._finalize = _finalize.__get__(mycc, mycc.__class__)
    return mycc



def BCCD(mf, frozen=None, u=None, conv_tol_normu=1e-5, max_cycle=20, diis=True,
         canonicalization=True):
    from pyscf.cc.bccd import bccd_kernel_
    from pyscf.lib import StreamObject
    mycc = CCSD(mf, frozen=frozen)

    class BCCD(mycc.__class__):
        def kernel(self):
            obj = self.view(mycc.__class__)
            obj.conv_tol = 1e-3
            obj.kernel()
            bccd_kernel_(obj, u, conv_tol_normu, max_cycle, diis,
                         canonicalization, self.verbose)
            self.__dict__.update(obj.__dict__)
            return self.e_tot

    return mycc.view(BCCD)


def FNOCCSD_kmp2(mf, thresh=1e-6, pct_occ=None, nvir_act=None, frozen=None, SS=None, chkfile=None):
    """Frozen natural orbital CCSD

    Attributes:
        thresh : float
            Threshold on NO occupation numbers. Default is 1e-6 (very conservative).
        pct_occ : float
            Percentage of total occupation number. Default is None. If present, overrides `thresh`.
        nvir_act : int
            Number of virtual NOs to keep. Default is None. If present, overrides `thresh` and `pct_occ`.
    """
    import numpy
    from pyscf import mp
    from pyscf.pbc import cc
    from pyscf.pbc.mp import kmp2
    import sys
    import pdb

    pt = kmp2.KMP2(mf, frozen=frozen).set(verbose=6).run()   
    no_energy, frozen, no_coeff = kmp2.KMP2.make_fno(pt, thresh=thresh, pct_occ=pct_occ, nvir_act=nvir_act)

    print(frozen)
    if len(frozen) == 0: frozen = None

    # coe = no_coeff
    # SS = mf.cell.intor("int1e_ovlp")

    # for idx, coe1 in enumerate(coe):
    #     matrix = numpy.conjugate(coe1.T) @ SS @coe1
    #     print(f'11111 is {matrix}')

    pt_no =kmp2.KMP2(mf, frozen= frozen, mo_coeff=no_coeff).set(verbose=5)
    pt_no.mo_energy = no_energy
    pt_no.kernel()
    delta_ekmp2 = pt.e_corr - pt_no.e_corr
    print(f'delta_ekmp2 is {delta_ekmp2}')

    mycc = cc.KCCSD(mf, frozen=frozen, mo_coeff=no_coeff) 

    mycc.delta_ekmp2 = pt.e_corr - pt_no.e_corr
    print('FNOCCSD: Delta E(krpa) =', mycc.delta_ekmp2)

    frozen_flat = numpy.concatenate(frozen)  # Flatten all arrays into one long array
    mycc.total_count = len(frozen_flat)
    print(f"Total number of frozen orbitals: {mycc.total_count}")
    mycc.frozen = frozen

    from pyscf.lib import logger
    def _finalize(self):
        '''Hook for dumping results and clearing up the object.'''
        if self.converged:
            logger.info(self, 'FNO-%s converged', self.__class__.__name__)
        else:
            logger.note(self, 'FNO-%s not converged', self.__class__.__name__)
        logger.note(self, 'E(FNO-%s) = %.16g  E_corr = %.16g',
                    self.__class__.__name__, self.e_tot, self.e_corr)
        logger.note(self, 'E(FNO-%s+delta-kmp2) = %.16g  E_corr = %.16g',
                    self.__class__.__name__,self.e_tot+self.delta_ekmp2,
                    self.e_corr+self.delta_ekmp2) 
        return self
    mycc._finalize = _finalize.__get__(mycc, mycc.__class__)
    return mycc
