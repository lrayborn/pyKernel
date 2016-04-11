# CESM analysis routines
import numpy as np
from climlab import constants as const
from climlab import thermo
import netCDF4 as nc
from scipy import integrate
import copy

def global_mean( zonfield, lat_deg ):
    '''Compute area-weighted global mean of a zonally averaged field.'''
    return sum(np.cos(np.deg2rad(lat_deg))*zonfield) / sum(np.cos(np.deg2rad(lat_deg)))

def inferred_heat_transport( energy_in, lat_deg ):
    '''Returns the inferred heat transport (in PW) by integrating the net energy imbalance from pole to pole.'''
    lat_rad = np.deg2rad( lat_deg )
    return ( 1E-15 * 2 * np.math.pi * const.a**2 * integrate.cumtrapz( np.cos(lat_rad)*energy_in,
            x=lat_rad, initial=0. ) )

def overturning(V, lat_deg, p_mb):
    '''compute overturning mass streamfunction (SV) from meridional velocity V, latitude (degrees) and pressure levels (mb)'''
    return 2*np.pi*const.a/const.g*np.cos(np.deg2rad(lat_deg))*integrate.cumtrapz(V,p_mb*const.mb_to_Pa,axis=0,initial=0)*1E-9


class run:
    def __init__(self, ncfile ):
        self.ncfile = ncfile
        self.ncdata = nc.Dataset( ncfile )
        # Grid information
        self.lat = self.ncdata.variables['lat'][:]
        self.lev = self.ncdata.variables['lev'][:]
        self.slat = np.sin(np.deg2rad(self.lat))
        self.biglat,self.p = np.meshgrid(self.lat,self.lev)
        self.dp,ignored = np.gradient(self.p)
        #  Load and calculate all the fields
        self.load_fields()
        self.do_diagnostics()
        
    def load_fields(self):
        '''Read fields directly from the netcdf model output file.'''
        self.ncfields = ['TS','T','Ta','OLR','ASR','OLRclr','ASRclr','cloud','CLDTOT','Q','relhum','omega','PS','U','V',
            'LHF','SHF','LWsfc','SWsfc','SnowFlux','Evap','Precip','U10','Z','SWdown_sfc','SWdown_sfc_clr','SWsfc_clr',
            'LWsfc_clr','SWdown_toa','TMQ']
        self.TS = np.squeeze(self.ncdata.variables['TS'][:])
        self.T = np.squeeze(self.ncdata.variables['T'][:])
        self.Ta = np.squeeze(self.ncdata.variables['TREFHT'][:])  # near-surface air temperature
        self.OLR = np.squeeze(self.ncdata.variables['FLNT'][:])
        self.ASR = np.squeeze(self.ncdata.variables['FSNT'][:])
        self.OLRclr = np.squeeze(self.ncdata.variables['FLNTC'][:])
        self.ASRclr = np.squeeze(self.ncdata.variables['FSNTC'][:])
        self.cloud = np.squeeze(self.ncdata.variables['CLOUD'][:])
        self.CLDTOT = np.squeeze(self.ncdata.variables['CLDTOT'][:])
        self.Q = np.squeeze(self.ncdata.variables['Q'][:])
        self.relhum = np.squeeze(self.ncdata.variables['RELHUM'][:])
        self.omega = np.squeeze(self.ncdata.variables['OMEGA'][:])
        self.PS = np.squeeze(self.ncdata.variables['PS'][:])
        self.U = np.squeeze(self.ncdata.variables['U'][:])
        self.V = np.squeeze(self.ncdata.variables['V'][:])
        #  surface energy budget terms, all defined as positive up (from ocean to atmosphere)
        self.LHF = np.squeeze(self.ncdata.variables['LHFLX'][:])
        self.SHF = np.squeeze(self.ncdata.variables['SHFLX'][:])
        self.LWsfc = np.squeeze(self.ncdata.variables['FLNS'][:])
        self.LWsfc_clr = np.squeeze(self.ncdata.variables['FLNSC'][:])
        self.SWsfc = -np.squeeze(self.ncdata.variables['FSNS'][:])
        self.SWsfc_clr = np.squeeze(self.ncdata.variables['FSNSC'][:])
        self.SnowFlux =  np.squeeze(self.ncdata.variables['PRECSC'][:]+self.ncdata.variables['PRECSL'][:])*const.rho_w*const.Lhfus
        #  more
        self.SWdown_sfc = np.squeeze(self.ncdata.variables['FSDS'][:])  # positive down
        self.SWdown_sfc_clr = np.squeeze(self.ncdata.variables['FSDSC'][:])
        self.SWdown_toa = np.squeeze(self.ncdata.variables['SOLIN'][:])
        #  hydrological cycle
        self.Evap = np.squeeze(self.ncdata.variables['QFLX'][:])  # kg/m2/s or mm/s
        self.Precip = np.squeeze(self.ncdata.variables['PRECC'][:]+self.ncdata.variables['PRECL'][:])*const.rho_w  # kg/m2/s or mm/s
        self.U10 = np.squeeze(self.ncdata.variables['U10'][:])  # near-surface wind speed
        self.Z = np.squeeze(self.ncdata.variables['Z3'][:])
        self.TMQ = np.squeeze(self.ncdata.variables['TMQ'][:])  # precipitable water in kg/m2
        
    def do_diagnostics(self):
        '''Compute a bunch of additional diagnostics.'''
        self.derivedfields = ['SST','TS_global','SST_global','Rtoa','dz','dTdp_moistadiabat','dTdp','dTdp_moistanom',
            'dTdz','dTdz_moistanom','w','Psi','V_imbal','V_bal','Psi_bal','SurfaceRadiation','SurfaceHeatFlux',
            'Fatmin','EminusP','HT_total','HT_atm','HT_ocean','HT_latent','HT_dse','EIS','DSE','MSE',
            'OLRcld','ASRcld','Rtoacld', 'LWsfc_cld','SWsfc_cld','SurfaceRadiation_clr','SurfaceRadiation_cld']
        self.SST = self.TS - const.tempCtoK
        self.TS_global = global_mean(self.TS,self.lat)
        self.SST_global = self.TS_global - const.tempCtoK
        self.Rtoa = self.ASR - self.OLR  # net downwelling radiation
        self.dz = -const.Rd * self.T / const.g * self.dp / self.p * 1.E-3  #  in km
        self.dTdp_moistadiabat = thermo.pseudoadiabat(self.T,self.p)
        self.dTdp,ignored = np.gradient(self.T) / self.dp
        self.dTdp_moistanom = self.dTdp - self.dTdp_moistadiabat
        self.dTdz = self.dTdp * self.dp / self.dz  # in K / km
        self.dTdz_moistanom = self.dTdp_moistanom * self.dp / self.dz
        # convert OMEGA (in Pa/s) to w (in m/s)
        self.w = -self.omega*const.Rd/const.g*self.T/(self.p*const.mb_to_Pa)
        # overturning mass streamfunction (in 10^9 kg/s or "mass Sverdrup")
        self.Psi = overturning(self.V,self.lat,self.lev)
        #  correct for mass imbalance....
        self.V_imbal = np.trapz(self.V/self.PS, self.lev*100, axis=0)
        self.V_bal = self.V - self.V_imbal
        self.Psi_bal = overturning(self.V_bal,self.lat,self.lev)
        self.SurfaceRadiation = self.LWsfc + self.SWsfc  # net upward radiation from surface
        self.SurfaceHeatFlux = self.SurfaceRadiation + self.LHF + self.SHF + self.SnowFlux  # net upward surface heat flux
        self.Fatmin = self.Rtoa + self.SurfaceHeatFlux  # net heat flux in to atmosphere
        self.EminusP = self.Evap - self.Precip  # kg/m2/s or mm/s
        # heat transport terms
        self.HT_total = inferred_heat_transport( self.Rtoa, self.lat )
        self.HT_atm = inferred_heat_transport( self.Fatmin, self.lat )
        self.HT_ocean = inferred_heat_transport( -self.SurfaceHeatFlux, self.lat )
        self.HT_latent = inferred_heat_transport( self.EminusP*const.Lhvap, self.lat ) # atm. latent heat transport from moisture imbal.
        self.HT_dse = self.HT_atm - self.HT_latent  # dry static energy transport as residual
        ind700 = np.nonzero(np.abs(self.lev-700)==np.min(np.abs(self.lev-700)))  # closest vertical level to 700 mb
        T700 = np.squeeze(self.T[ind700,:])
        self.EIS = thermo.EIS(self.Ta,T700)
        self.DSE = const.cp * self.T + const.g * self.Z
        self.MSE = self.DSE + const.Lhvap * self.Q #  J / kg
        self.ASRcld = self.ASR - self.ASRclr
        self.OLRcld = self.OLR - self.OLRclr
        self.Rtoacld = self.ASRcld - self.OLRcld
        self.LWsfc_cld = self.LWsfc - self.LWsfc_clr  # all the surface radiation terms are defined positive up
        self.SWsfc_cld = self.SWsfc - self.SWsfc_clr
        self.SurfaceRadiation_clr = self.LWsfc_clr + self.SWsfc_clr  
        self.SurfaceRadiation_cld = self.SurfaceRadiation - self.SurfaceRadiation_clr
        #  
        
    def __sub__(self, other):
        #  create a new CESM.run object
        diff = run(self.ncfile)
        # populate it with differences
        for item in self.ncfields + self.derivedfields:
            exec('diff.' + item + ' = self.' + item + ' - other.' + item )
        return diff
   
class GRaMrun( run ):
    def __init__(self, ncfile ):
        self.ncfile = ncfile
        self.ncdata = nc.Dataset( ncfile )
        # Grid information
        self.lat = self.ncdata.variables['lat'][:]
        self.lev = self.ncdata.variables['pfull'][:]
        self.slat = np.sin(np.deg2rad(self.lat))
        self.biglat,self.p = np.meshgrid(self.lat,self.lev)
        self.dp,ignored = np.gradient(self.p)
        #  Load and calculate all the fields
        self.load_fields()
        self.do_diagnostics()
        
    def __sub__(self, other):
        #  create a new CESM.GRaMrun object
        diff = GRaMrun(self.ncfile)
        # populate it with differences
        for item in self.ncfields + self.derivedfields:
            exec('diff.' + item + ' = self.' + item + ' - other.' + item )
        return diff

    def load_fields(self):
        '''Read fields directly from the netcdf model output file.'''
        self.ncfields = ['TS','T','Ta','OLR','ASR','ASRclr','OLRclr','cloud','Q','relhum','omega','PS','U','V','LHF','SHF',
            'LWsfc','SWsfc','SnowFlux','Evap','Precip','U_star','U10','Z','SWsfc_clr',
            'LWsfc_clr','TMQ']
        self.TS = np.squeeze(self.ncdata.variables['t_surf'][:])
        self.T = np.squeeze(self.ncdata.variables['temp'][:])
        self.Ta = thermo.potential_temperature(self.T[-1,:],self.lev[-1])  # use lowest model level, convert to surface value
        #self.Ta = np.squeeze(self.ncdata.variables['t_ref'][:])  # near-surface air temperature
        self.OLR = np.squeeze(self.ncdata.variables['olr'][:])
        self.ASR = -np.squeeze(self.ncdata.variables['flux_sw'][0,0,:])
        self.OLRclr = self.OLR; self.ASRclr = self.ASR
        self.cloud = np.zeros_like(self.T)
        self.Q = np.squeeze(self.ncdata.variables['sphum'][:])
        self.relhum = np.squeeze(self.ncdata.variables['rh'][:])
        self.omega = np.squeeze(self.ncdata.variables['omega'][:])
        self.PS = np.squeeze(self.ncdata.variables['ps'][:])
        self.U = np.squeeze(self.ncdata.variables['ucomp'][:])
        self.V = np.squeeze(self.ncdata.variables['vcomp'][:])
        #  surface energy budget terms, all defined as positive up (from ocean to atmosphere)
        self.LHF = np.squeeze(self.ncdata.variables['evap'][:])*const.Lhvap
        self.SHF = np.squeeze(self.ncdata.variables['shflx'][:])
        self.LWsfc = -np.squeeze(self.ncdata.variables['lwflx'][:])
        self.SWsfc = self.ASR  # GRaM is transparent to shortwave!
        self.SnowFlux =  np.zeros_like(self.LHF)  # no snow in GRaM!
        #  hydrological cycle
        self.Evap = np.squeeze(self.ncdata.variables['evap'][:])  # kg/m2/s or mm/s
        self.Precip = np.squeeze(self.ncdata.variables['precip'][:])  # kg/m2/s or mm/s
        self.U_star = np.squeeze(self.ncdata.variables['u_star'][:])  # near-surface wind speed
        self.U10 = self.U_star  # for now...
        dse = np.squeeze(np.mean(self.ncdata.variables['dry_stat_en'][:],axis=3))
        self.Z = ( dse - const.cp * self.T ) / const.g
        self.TMQ = np.zeros_like(self.OLR)  # precipitable water in kg/m2... will need to integrate specific humidity to get this.
        # clear-sky surface fluxes
        self.LWsfc_clr = self.LWsfc; self.SWsfc_clr = self.SWsfc

class AM2run( GRaMrun ):
    def __sub__(self, other):
        #  create a new CESM.AM2run object
        diff = AM2run(self.ncfile)
        # populate it with differences
        for item in self.ncfields + self.derivedfields:
            exec('diff.' + item + ' = self.' + item + ' - other.' + item )
        return diff
        
    def load_fields(self):
        '''Read fields directly from the netcdf model output file.'''
        self.ncfields = ['TS','T','Ta','OLR','ASR','OLRclr','ASRclr','cloud','Q','relhum','omega','PS','U','V','LHF','SHF',
            'LWsfc','SWsfc','SnowFlux','Evap','Precip','U10','Z','SWdown_sfc','SWdown_sfc_clr','SWsfc_clr',
            'LWsfc_clr','SWdown_toa','TMQ']
        self.TS = np.squeeze(self.ncdata.variables['t_surf'][:])
        self.T = np.squeeze(self.ncdata.variables['temp'][:])
        #self.Ta = thermo.potential_temperature(self.T[-1,:],self.lev[-1])  # use lowest model level, convert to surface value
        self.Ta = np.squeeze(self.ncdata.variables['t_ref'][:])  # near-surface air temperature
        self.OLR = np.squeeze(self.ncdata.variables['olr'][:])
        self.ASR = np.squeeze(self.ncdata.variables['swdn_toa'][:]-self.ncdata.variables['swup_toa'][:])
        self.OLRclr = np.squeeze(self.ncdata.variables['olr_clr'][:])
        self.ASRclr = np.squeeze(self.ncdata.variables['swdn_toa_clr'][:]-self.ncdata.variables['swup_toa_clr'][:])
        self.cloud = np.squeeze(self.ncdata.variables['cld_amt_dyn'][:])
        self.Q = np.squeeze(self.ncdata.variables['sphum'][:])
        self.relhum = np.squeeze(self.ncdata.variables['rh'][:])
        self.omega = np.squeeze(self.ncdata.variables['omega'][:])
        self.PS = np.squeeze(self.ncdata.variables['ps'][:])
        self.U = np.squeeze(self.ncdata.variables['ucomp'][:])
        self.V = np.squeeze(self.ncdata.variables['vcomp'][:])
        #  surface energy budget terms, all defined as positive up (from ocean to atmosphere)
        self.LHF = np.squeeze(self.ncdata.variables['evap'][:])*const.Lhvap
        self.SHF = np.squeeze(self.ncdata.variables['shflx'][:])
        self.LWsfc = -np.squeeze(self.ncdata.variables['lwflx'][:])
        self.SWsfc = np.squeeze(self.ncdata.variables['swup_sfc'][:]-self.ncdata.variables['swdn_sfc'][:])
        self.SnowFlux =  np.squeeze(self.ncdata.variables['snow_ls'][:]+self.ncdata.variables['snow_conv'][:])*const.Lhfus
        #  hydrological cycle
        self.Evap = np.squeeze(self.ncdata.variables['evap'][:])  # kg/m2/s or mm/s
        self.Precip = np.squeeze(self.ncdata.variables['precip'][:])  # kg/m2/s or mm/s
        self.U10 = np.squeeze(self.ncdata.variables['wind'][:])  # near-surface wind speed
        self.Z = np.squeeze(self.ncdata.variables['z_full'][:])
        self.TMQ = np.squeeze(self.ncdata.variables['WVP'][:])  # precipitable water in kg/m2
        # clear-sky surface fluxes
        self.SWdown_sfc = np.squeeze(self.ncdata.variables['swdn_sfc'][:])
        self.SWdown_sfc_clr = np.squeeze(self.ncdata.variables['swdn_sfc_clr'][:])
        self.SWsfc_clr = np.squeeze(self.ncdata.variables['swup_sfc_clr'][:]-self.ncdata.variables['swdn_sfc_clr'][:])
        self.LWsfc_clr = np.squeeze(self.ncdata.variables['lwup_sfc_clr'][:]-self.ncdata.variables['lwdn_sfc_clr'][:])
        self.SWdown_toa = np.squeeze(self.ncdata.variables['swdn_toa'][:])

def fractional_change( pert, ctrl ):
    '''Compute fractional difference (difference of logs) of each field in the two CESM.run instances.'''
    fracdiff = copy.copy(ctrl)
    for item in fracdiff.ncfields + fracdiff.derivedfields:
        exec('fracdiff.' + item + ' = np.log(pert.' + item + ') - np.log(ctrl.' + item +')' )
    return fracdiff


class APRP:
    def __init__(self, ncdata_pert, ncdata_ctrl ):
        # quick hack to see what format the data are in
        if 'FSNS' in ncdata_ctrl.variables:
            self.pert = APRPterms(ncdata_pert)
            self.ctrl = APRPterms(ncdata_ctrl)
        elif 'swdn_toa' in ncdata_ctrl.variables:
            self.pert = APRPterms_AM2(ncdata_pert)
            self.ctrl = APRPterms_AM2(ncdata_ctrl)
        else:
            raise ValueError('Bad input!')
        
        self.params = ['c','alpha_clr','alpha_oc','mu_clr','mu_cld','gamma_clr','gamma_cld']
        self.dA = {}
        #  calculate all the single parameter terms
        for thisparam in self.params:
            self.dA[thisparam] = self.compute_partial_dA(self.pert,self.ctrl,thisparam)
        #  add terms together
        self.dA['alpha'] = self.dA['alpha_clr'] + self.dA['alpha_oc']
        self.dA['cld'] = self.dA['mu_cld'] + self.dA['gamma_cld'] + self.dA['c']
        self.dA['clr'] = self.dA['mu_clr'] + self.dA['gamma_clr']
        #  express in terms of changes in absorbed shortwave radiation in W/m2
        self.dASR = {}
        for param,delta_alb in self.dA.items():
            self.dASR[param] = - delta_alb * self.ctrl.insolation

    def compute_partial_dA( self, pert, ctrl, thisparam ):
        # make a dictionary of parameters to use for calculating albedo
        p_pert = {}; p_ctrl = {}
        for param in self.params:
            if param is thisparam:  # this is the parameter we are perturbing
                p_pert[param] = getattr(pert,param)
                p_ctrl[param] = getattr(ctrl,param)
            else:  #  use the average value
                p_ave = (getattr(pert,param)+getattr(ctrl,param)) / 2
                p_pert[param] = p_ave
                p_ctrl[param] = p_ave
        return self.Albedo( **p_pert ) - self.Albedo( **p_ctrl )
            
    def Albedo_perstream( self, mu, gamma, alpha ):
        '''Compute A (planetary albedo) for a single stream (clear sky or overcast) from radiative parameters.'''
        return mu*gamma + mu*alpha*(1-gamma)**2 / (1-alpha*gamma)
        
    def mu_oc( self, mu_clr, mu_cld ):
        return mu_clr * mu_cld
        
    def gamma_oc( self, gamma_clr, gamma_cld ):
        return 1 - (1-gamma_clr) * (1-gamma_cld)
        
    def Albedo( self, c, alpha_clr, alpha_oc, mu_clr, mu_cld, gamma_clr, gamma_cld ):
        A_clr = self.Albedo_perstream( mu_clr, gamma_clr, alpha_clr )
        A_oc = self.Albedo_perstream( self.mu_oc(mu_clr, mu_cld), self.gamma_oc(gamma_clr,gamma_cld), alpha_oc )
        return (1-c)*A_clr + c*A_oc


class APRPterms:
    '''Implements the Approximate Partial Radiative Perturbation feedback analysis, following Taylor et al., J. Climate 2007'''
    def __init__( self, ncdata ):
        self.insolation = np.squeeze(ncdata.variables['SOLIN'][:])
        self.c = np.squeeze(ncdata.variables['CLDTOT'][:])
        self.SWdown_sfc_clr = np.squeeze(ncdata.variables['FSDSC'][:])
        self.SWnet_sfc_clr = np.squeeze(ncdata.variables['FSNSC'][:])
        self.SWnet_toa_clr = np.squeeze(ncdata.variables['FSNTC'][:])
        self.SWdown_sfc_oc = self.oc_flux( np.squeeze(ncdata.variables['FSDS'][:]), 
                np.squeeze(ncdata.variables['FSDSC'][:]), self.c )
        self.SWnet_sfc_oc = self.oc_flux( np.squeeze(ncdata.variables['FSNS'][:]), 
                np.squeeze(ncdata.variables['FSNSC'][:]), self.c )
        self.SWnet_toa_oc = self.oc_flux( np.squeeze(ncdata.variables['FSNT'][:]),
                np.squeeze(ncdata.variables['FSNTC'][:]), self.c )
        self.setup_terms()
   
    def setup_terms(self):
        self.SWup_sfc_clr = self.SWdown_sfc_clr - self.SWnet_sfc_clr
        self.SWup_toa_clr = self.insolation - self.SWnet_toa_clr
        self.SWup_sfc_oc = self.SWdown_sfc_oc - self.SWnet_sfc_oc
        self.SWup_toa_oc = self.insolation - self.SWnet_toa_oc
        self.A_clr, self.alpha_clr, self.Qshat_clr = self.compute_fluxes( self.insolation, self.SWdown_sfc_clr, 
                self.SWup_sfc_clr, self.SWup_toa_clr )
        self.A_oc, self.alpha_oc, self.Qshat_oc = self.compute_fluxes( self.insolation, self.SWdown_sfc_oc, 
                self.SWup_sfc_oc, self.SWup_toa_oc )
        self.mu_clr = self.mu( self.A_clr, self.alpha_clr, self.Qshat_clr )
        mu_oc = self.mu( self.A_oc, self.alpha_oc, self.Qshat_oc )
        self.mu_cld = mu_oc / self.mu_clr
        self.gamma_clr = self.gamma( self.mu_clr, self.alpha_clr, self.Qshat_clr )
        gamma_oc = self.gamma( mu_oc, self.alpha_oc, self.Qshat_oc )
        self.gamma_cld = 1 - (1-gamma_oc)/(1-self.gamma_clr)
        
    def oc_flux( self, Ftotal, Fclr, c ):
        return (Ftotal - (1-c)*Fclr) / c
        
    def compute_fluxes( self, SWdown_toa, SWdown_sfc, SWup_sfc, SWup_toa ):
        A = SWup_toa / SWdown_toa
        alpha = SWup_sfc / SWdown_sfc
        Qshat = SWdown_sfc / SWdown_toa
        return A, alpha, Qshat
        
    def mu( self, A, alpha, Qshat ):
        return A + Qshat * (1-alpha)
        
    def gamma( self, mu, alpha, Qshat ):
        return ( mu - Qshat) / ( mu - alpha * Qshat )
        
        
class APRPterms_AM2( APRPterms ):
    '''Implements the Approximate Partial Radiative Perturbation feedback analysis, following Taylor et al., J. Climate 2007'''
    def __init__( self, ncdata ):
        self.insolation = np.squeeze(ncdata.variables['swdn_toa'][:])
        self.c = np.squeeze(ncdata.variables['tot_cld_amt'][:])/100.  # convert from percent to fraction
        self.SWdown_sfc_clr = np.squeeze(ncdata.variables['swdn_sfc_clr'][:])
        self.SWnet_sfc_clr = np.squeeze(ncdata.variables['swdn_sfc_clr'][:]-ncdata.variables['swup_sfc_clr'][:])
        self.SWnet_toa_clr = np.squeeze(ncdata.variables['swdn_toa'][:]-ncdata.variables['swup_toa_clr'][:])
        self.SWdown_sfc_oc = self.oc_flux( np.squeeze(ncdata.variables['swdn_sfc'][:]), 
                np.squeeze(ncdata.variables['swdn_sfc_clr'][:]), self.c )
        self.SWnet_sfc_oc = self.oc_flux( np.squeeze(ncdata.variables['swdn_sfc'][:]-ncdata.variables['swup_sfc'][:]), 
                np.squeeze(ncdata.variables['swdn_sfc_clr'][:]-ncdata.variables['swup_sfc_clr'][:]), self.c )
        self.SWnet_toa_oc = self.oc_flux( np.squeeze(ncdata.variables['swdn_toa'][:]-ncdata.variables['swup_toa'][:]),
                np.squeeze(ncdata.variables['swdn_toa'][:]-ncdata.variables['swup_toa_clr'][:]), self.c )
        self.setup_terms()


#  Experimenting with the new iris package
#  here is some example code to use iris to read in the CAM4 heat uptake runs
#   and make a simple plot of surface temperature anomalies

##  this code doesn't belong here but ...
#import iris
#path = '/Users/Brian/Documents/School Work/QAquMIP/model_runs/CAM4/'
#endstr = '.cam.h0.zonclim.nc'
#runs = ['ctrl','CO2','qupH','qupT']
#filelist = []
#for r in runs: filelist.append(path + 'QAqu_' + r + endstr)
#TS = iris.load(filelist, 'Surface temperature (radiative)')
#for index, item in enumerate(TS):
#    TS[index] = item.collapsed('time', iris.analysis.MEAN)
#TS_anom = iris.cube.CubeList()
#for thisTS in TS:
#    TS_anom.append(thisTS - TS[0])
#for index, item in enumerate(TS_anom):
#    iris.quickplot.plot(item, label=runs[index])
#gca().legend()