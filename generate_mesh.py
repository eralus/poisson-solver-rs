import numpy as np
from scipy.sparse import coo_array
import pyvista as pv
import scipy.linalg as spla
from scipy.sparse.linalg import spsolve, gmres
from scipy.sparse import csc_array, diags
from numpy.linalg import solve
import multiprocessing as mp
from numbers import Number

class PoissonSolver:
    def __init__(self,
                 pv_tetra_mesh:pv.UnstructuredGrid,
                 wf_vals:np.ndarray,
                 is_dirichlet:np.ndarray,
                 dirichlet_vals:np.ndarray,
                 eps:np.ndarray=None,
                 wf_potentials:np.ndarray=None,
                 temp:np.double=300,
                 doping_density=None):       # doping_density is positive when there are electrons missing
        self.eps0 = 8.854e-12 * 1e-10 #Farad/Angstrom
        self.elem_charge = 1.602176634e-19 #Coulomb
        self.kB = 1.380649e-23 #Joule/Kelvin
        self.temp = temp
        self.cells = np.copy(pv_tetra_mesh.cells.reshape((-1,5))[:,1:])
        self.points = np.copy(pv_tetra_mesh.points)
        self.cell_vols = pv_tetra_mesh.compute_cell_sizes()['Volume']
        self.wfs = np.copy(wf_vals)
        self.is_dirichlet = np.copy(is_dirichlet)
        self.dirichlet_vals = np.copy(dirichlet_vals)
        self.rho = np.zeros(np.sum(1-self.is_dirichlet).astype(np.int32))
        if doping_density is None:
            self.doping_density = np.zeros(np.sum(1-self.is_dirichlet).astype(np.int32))
        if isinstance(doping_density, np.ndarray):
            if doping_density.shape[0] == 2:
                self.doping_density=np.zeros(np.sum(1-self.is_dirichlet).astype(np.int32))
                left = (max(self.points[:,0])-min(self.points[:,0]))/3.0 + min(self.points[:,0])
                right = 2.0*(max(self.points[:,0])-min(self.points[:,0]))/3.0 + min(self.points[:,0])
                self.doping_density[self.points[:,0] < left] = doping_density[0]*self.elem_charge
                self.doping_density[self.points[:,0] > right] = doping_density[1]*self.elem_charge
                self.doping_density_from_init = np.copy(doping_density)
            else:
                assert(doping_density.shape[0] == (self.points.shape[0]))
                self.doping_density = np.copy(doping_density)[is_dirichlet==0]
                self.doping_density *= self.elem_charge
        else:
            assert(isinstance(doping_density, Number))
            # left = min(self.points[:,0])+10.
            # right = max(self.points[:,0])-10.
            left = (max(self.points[:,0])-min(self.points[:,0]))/3.0 + min(self.points[:,0])
            right = 2.0*(max(self.points[:,0])-min(self.points[:,0]))/3.0 + min(self.points[:,0])
            is_doped = abs((self.points[:,0]<right).astype(np.int8) - (self.points[:,0]>left).astype(np.int8))[self.is_dirichlet == 0]
            
            # self.doping_density = np.ones(np.sum(is_dirichlet == 0)) * doping_density
            self.doping_density = is_doped*doping_density*self.elem_charge
        self.calculate_volume_per_point()
        print("calculated integration volumes per point")
        self.normalize_wfs()
        print("normalized wfs")
        self.map_var_pts()
        self.set_permittivity(eps)
        if wf_potentials is None:
            # self.phi = ((self.points[self.is_dirichlet==0,0]-min(self.points[:,0]))/(max(self.points[:,0])-min(self.points[:,0])) *
            #             (max(self.dirichlet_vals)-min(self.dirichlet_vals)) + min(self.dirichlet_vals))
            self.phi = np.zeros(np.sum(self.is_dirichlet == 0))
            self.project_potential_on_wfs(self.phi)
        else:
            assert(wf_potentials.shape[0] == self.wfs.shape[0])
            self.wf_potentials = np.copy(wf_potentials)
            self.phi = (np.sum(self.wf_potentials.reshape((-1,1)) * self.wfs**2, axis=0) / np.sum(self.wfs**2, axis=0))[self.is_dirichlet==0]
        self.prev_wf_potentials = np.copy(self.wf_potentials)
        self.initialize_matrix()
        print("initialized stiffness matrix")
        
    def normalize_wfs(self):
        # wf_vols is not useful within this class. It's only useful for testing the quality of different sampling methods. Should be removed at some point
        self.wf_vols = np.sum(self.wfs**2 * self.tot_vols, axis=1).reshape((-1,1))
        print(min(self.tot_vols))
        # print( np.sqrt(np.sum(self.wfs**2 * self.tot_vols, axis=1)).reshape((-1,1)))
        self.wfs /= np.sqrt(np.sum(self.wfs**2 * self.tot_vols, axis=1)).reshape((-1,1))
        self.wf_centers = np.sum((self.wfs**2 * self.tot_vols).reshape(self.wfs.shape[0],self.points.shape[0],1) * self.points.reshape((1,-1,3)), axis=1)
        print(self.wf_centers.shape)
        print([self.wfs.shape[0], 3])
    
    def map_var_pts(self):
        self.variable_pts = np.zeros(self.points.shape[0], dtype=np.int32)
        var_idx = 0
        for point_idx in range(self.points.shape[0]):
            if self.is_dirichlet[point_idx]:
                self.variable_pts[point_idx] = -1
            else:
                self.variable_pts[point_idx] = var_idx
                var_idx += 1
        self.n_var_pts = var_idx
        

    def initialize_matrix(self):
        with mp.Pool() as pool:
            results = pool.map(_initialize_matrix_helper, [(self.points[self.cells[cell_idx]],
                                                           self.cell_vols[cell_idx],
                                                           self.variable_pts[self.cells[cell_idx]],
                                                           self.dirichlet_vals[self.cells[cell_idx]],
                                                           self.eps[cell_idx])
                                                           for cell_idx in range(self.cells.shape[0])])
        
        results = np.array(results)
        s_data = results[:,0,:].reshape(-1)
        s_rows = results[:,1,:].reshape(-1).astype(np.int32)
        s_cols = results[:,2,:].reshape(-1).astype(np.int32)
        r_data = results[:,3,:].reshape(-1)
        r_rows = results[:,4,:].reshape(-1).astype(np.int32)
        r_cols = results[:,5,:].reshape(-1).astype(np.int32)

        self.stiffness_mat = coo_array((s_data[s_cols != -1], (s_rows[s_cols != -1], s_cols[s_cols != -1])), shape=(self.n_var_pts, self.n_var_pts)).tocsc()
        
        self.dirichlet_term = np.zeros(self.n_var_pts)
        # print("no dirichlet")
        s_data = s_data[s_cols == -1]
        s_rows = s_rows[s_cols == -1]
        for idx in range(s_data.shape[0]):
            self.dirichlet_term[s_rows[idx]] += s_data[idx]
        
        self.rho_mat = coo_array((r_data, (r_rows, r_cols)), shape=(self.n_var_pts, self.n_var_pts)).tocsc()

    
    def set_charge_densities(self, electron_density:np.ndarray, orbital_doping=None):
        self.separate_charges = False
        if electron_density.ndim == 1:
            assert(electron_density.shape[0] == self.wfs.shape[0])
            print("diagonal densities")
            self.rho = (self.wfs**2 * electron_density.reshape((-1,1))).sum(axis=0) * -1 *self.elem_charge
            self.wf_el_densities = np.copy(electron_density)
        elif electron_density.ndim == 2:
            assert(electron_density.shape == (self.wfs.shape[0],self.wfs.shape[0]))
            print("offdiagonal densities")
            self.rho = np.real(((electron_density@self.wfs) * self.wfs).sum(axis=0) * -1 * self.elem_charge)
            self.wf_el_densities = np.real(np.diag(electron_density))
            self.charge_density_matrix = np.copy(electron_density)
        elif electron_density.ndim == 3:
            # electron_density[0] is electrons, electron_density[1] is holes
            assert(electron_density.shape == (2,self.wfs.shape[0],self.wfs.shape[0]))
            assert(np.all(np.real(np.diag(electron_density[0,:,:]))>=0.0))
            assert(np.all(np.real(np.diag(electron_density[1,:,:]))>=0.0))
            cumul_density = electron_density[0,:,:]-electron_density[1,:,:]
            self.rho = np.real(((cumul_density@self.wfs) * self.wfs).sum(axis=0) * -1 * self.elem_charge)
            self.wf_el_densities = np.real(np.array([np.diag(electron_density[0,:,:]),np.diag(electron_density[1,:,:])]))
            self.charge_density_matrix = np.copy(electron_density)
            self.separate_charges = True
        else:
            raise Exception("electron_density needs to be either vector of rhos, or the G_lesser matrix")
        self.rho = self.rho[self.is_dirichlet == 0]
        if orbital_doping is not None:
            # is_doped = 1 - (np.isclose(self.doping_density,0.)).astype(np.int8)
            # vol = np.sum(self.tot_vols[self.is_dirichlet==0][is_doped])
            # self.doping_density[is_doped] = sum(electron_density)/vol * self.elem_charge
            # tot_doping = np.sum(self.tot_vols * self.doping_density/self.elem_charge)
            # self.doping_density *= sum(electron_density)/tot_doping * self.elem_charge
            # doping_per_atom = sum(electron_density) / 128.
            # doping_per_atom = tot_doping/256.
            orbital_doping = orbital_doping.astype(np.int8)
            assert(orbital_doping.shape[0] == 2)
            if hasattr(self, "doping_density_from_init"):
                self.doping_density *= 0.
                self.doping_density += np.sum((self.wfs**2)[:orbital_doping[0]*32,:], axis=0)*self.doping_density_from_init[0]*self.elem_charge
                self.doping_density += np.sum((self.wfs**2)[-orbital_doping[1]*32:,:], axis=0)*self.doping_density_from_init[1]*self.elem_charge
            else:
                doping_per_atom = self.doping_density[np.argmax(abs(self.doping_density))]/self.elem_charge
                self.doping_density *= 0.
                self.doping_density += np.sum((self.wfs**2)[:orbital_doping[0]*32,:] * doping_per_atom, axis=0)*self.elem_charge
                self.doping_density += np.sum((self.wfs**2)[-orbital_doping[1]*32:,:] * doping_per_atom, axis=0)*self.elem_charge

    def solve_poisson(self):
        self.phi = spsolve(self.stiffness_mat, self.dirichlet_term - self.rho + self.doping_density)
        self.project_potential_on_wfs(self.phi)
        return np.copy(self.wf_potentials)
    
    def solve_poisson_NR(self, factor=1.0, finite_diff=True, drho_dV=None, mixing=False):
        # if a_posteriori and hasattr(self, "prev_wf_el_densities") and hasattr(self, "prev_wf_potentials"):
        if finite_diff and (hasattr(self, "prev_wf_el_densities") or drho_dV is not None):
            # assert(hasattr(self, "drho_dV_1D"))
            if drho_dV is None:
                self.drho_dV_1D = (self.wf_el_densities-self.prev_wf_el_densities)/(self.wf_potentials-self.prev_wf_potentials)
            else:
                self.drho_dV_1D = drho_dV
            self.drho_dV_from_1D()
            A = self.stiffness_mat - self.rho_mat @ csc_array(self.drho_dV)
            # d_phi = spsolve(-A, self.stiffness_mat@self.phi - self.rho_mat@(self.rho+self.doping_density)-self.dirichlet_term)
            d_phi = solve(-A.toarray(), self.stiffness_mat@self.phi - self.rho_mat@(self.rho+self.doping_density)-self.dirichlet_term)
            print("solved")
            self.prev_wf_potentials = np.copy(self.wf_potentials)
            self.prev_phi = np.copy(self.phi)
            self.phi += factor*d_phi
            d_wf_pot = -np.copy(self.wf_potentials)
            self.project_potential_on_wfs(self.phi)
            d_wf_pot += self.wf_potentials
            print("max d_wf_V:", max(abs(d_wf_pot)))
            print("min d_wf_V:", min(abs(d_wf_pot)))
            np.save("potential.npy", self.phi)
            self.prev_wf_el_densities = np.copy(self.wf_el_densities)
            return np.copy(self.wf_potentials)

        counter = 0
        self.find_fermi()
        d_wf_pot = -np.copy(self.wf_potentials)
        prev_phi = np.copy(self.phi)
        while(counter < 10):
            self.calculate_drho_dV()
            A = self.stiffness_mat - self.rho_mat @ csc_array(self.drho_dV)
            # M = diags(1./A.diagonal(), format="csc")
            # d_phi,info = gmres(-A, self.stiffness_mat@self.phi + (self.rho+self.doping_density-self.dirichlet_term), M=-M)
            # print(info)
            d_phi = solve(-A.toarray(), self.stiffness_mat@self.phi - self.rho_mat@(self.rho+self.doping_density)-self.dirichlet_term)
            print("solved")
            self.phi += factor*d_phi
            d_wf_pot = -self.wf_potentials
            self.project_potential_on_wfs(self.phi)
            d_wf_pot += self.wf_potentials
            if not self.separate_charges:
                self.wf_el_densities = (self.rhsign * self.NC * np.log(1 + np.exp(
                    self.rhsign*self.elem_charge*(self.wf_fermi_energies - self.wf_potentials)/(self.kB*self.temp))))
                if hasattr(self, "charge_density_matrix"):
                    np.fill_diagonal(self.charge_density_matrix, self.wf_el_densities)
                    self.set_charge_densities(self.charge_density_matrix)
                else:
                    self.set_charge_densities(self.wf_el_densities)
            else:
                self.wf_el_densities = self.NC.reshape((2,1)) * np.log(1 + np.exp(
                    np.array([[-1],[1]])*self.elem_charge*(self.wf_fermi_energies - self.wf_potentials)/(self.kB*self.temp)))
                np.fill_diagonal(self.charge_density_matrix[0,:,:],self.wf_el_densities[0,:])
                np.fill_diagonal(self.charge_density_matrix[1,:,:],self.wf_el_densities[1,:])
                self.set_charge_densities(self.charge_density_matrix)

            # self.rho += self.drho_dV@d_phi
            # self.wf_el_densities += np.diag(self.drho_dV_1D) @ (self.tot_vols*self.wfs**2)[:,self.is_dirichlet==0] @ d_phi
            # if max(abs(d_phi)) < 1e-8:
            if max(abs(d_wf_pot)) < 1e-3:
                print("max abs(d_wf_V) before abort:", max(abs(d_wf_pot)))
                break
            print("max abs(d_phi):", max(abs(d_phi)))
            print("max V", max(self.phi))
            print("min V", min(self.phi))
            print("max wf_V", max(self.wf_potentials))
            print("min wf_V", min(self.wf_potentials))
            print("min abs(V)", min(abs(self.phi)))
            print("median d_el_dens", np.median(np.diag(self.drho_dV_1D) @ (self.tot_vols*self.wfs**2)[:,self.is_dirichlet==0] @ d_phi))
            print("median el_density", np.median(self.wf_el_densities))
            counter += 1
        if mixing:
            self.wf_potentials = 0.5*self.wf_potentials - 0.5*d_wf_pot
            self.phi = 0.5*self.phi + 0.5*prev_phi
        d_wf_pot += self.wf_potentials
        print("max d_wf_V:", max(abs(d_wf_pot)))
        print("min d_wf_V:", min(abs(d_wf_pot)))
        np.save("potential.npy", self.phi)
        self.prev_wf_el_densities = np.copy(self.wf_el_densities)
        return np.copy(self.wf_potentials)


    def project_potential_on_wfs(self, potential:np.ndarray):
        assert(potential.shape[0] == self.points.shape[0] or potential.shape[0] == sum(self.is_dirichlet==0))
        if(potential.shape == sum(self.is_dirichlet==0)):
            potential = np.zeros(self.points.shape[0])
            potential[self.is_dirichlet==0] = self.phi
        self.wf_potentials = np.sum(self.wfs**2 * potential * self.tot_vols, axis=1)
    
    def calculate_volume_per_point(self):
        self.tot_vols = np.zeros(self.points.shape[0])
        for cell_idx in range(self.cells.shape[0]):
            self.tot_vols[self.cells[cell_idx,:]] += self.cell_vols[cell_idx]
        self.tot_vols /= 4.0
        self.tot_vols[self.is_dirichlet != 0] = 0
        
    def find_fermi(self):
        if not hasattr(self, "wf_fermi_energies"): self.wf_fermi_energies = np.zeros_like(self.wf_el_densities)
        if not self.separate_charges:
            # self.NC = np.max([np.max(abs(self.wf_el_densities)),1e-6])
            self.NC = np.max(abs(self.wf_el_densities))
            self.rhsign = np.sign(self.wf_el_densities)
            self.rhsign[np.isclose(self.rhsign, 0)] = 1
            drho = 1e-8
            self.wf_fermi_energies = (self.wf_potentials 
                                    + self.rhsign*self.kB*self.temp/self.elem_charge
                                    * np.log(np.exp((self.wf_el_densities+self.rhsign*drho)/(self.NC*self.rhsign))-1))
        else:
            # self.NC = np.max([np.max(abs(self.wf_el_densities),axis=1),1e-6],axis=0)
            self.NC=np.zeros(2)
            self.NC[0] = np.max([np.max(abs(self.wf_el_densities[0,:])),1e-6])
            self.NC[1] = np.max([np.max(abs(self.wf_el_densities[1,:])),1e-6])
            print('self.NC: ', self.NC)
            # self.rhsign = np.sign(self.wf_el_densities)
            # self.rhsign[np.isclose(self.rhsign, 0)] = 1
            drho = 1e-8
            self.wf_fermi_energies = (self.wf_potentials
                                    + np.array([[-1],[1]])*self.kB*self.temp/self.elem_charge
                                    * np.log(np.exp((self.wf_el_densities+drho)/(self.NC.reshape((2,1))))-1))
            # print(np.log(np.exp((self.wf_el_densities+drho)/(self.NC.reshape((2,1))))-1))
        
    def calculate_drho_dV(self):
        # rhsign = 1.
        # self.find_fermi()
        if not self.separate_charges:
            self.drho_dV_1D = (-self.elem_charge*self.NC/(self.kB*self.temp)/
                        (np.exp(self.rhsign*self.elem_charge*(self.wf_potentials-self.wf_fermi_energies)/(self.kB*self.temp))+1))
        else:
            self.drho_dV_1D = (-self.elem_charge*self.NC.reshape((2,1))/(self.kB*self.temp)/
                        (np.exp(np.array([[-1],[1]])*self.elem_charge*(self.wf_potentials-self.wf_fermi_energies)/(self.kB*self.temp))+1)).sum(axis=0)
        self.drho_dV_from_1D()
    
    def drho_dV_from_1D(self):
        self.drho_dV = (np.transpose((self.wfs**2)[:,self.is_dirichlet==0]*-1*self.elem_charge)@ 
                        np.diag(self.drho_dV_1D) @ 
                       (self.tot_vols*self.wfs**2)[:,self.is_dirichlet==0])

    def set_permittivity(self, eps):
        # eps is None -> eps is 1.0 everywhere
        # eps is 1D -> scalar -> isotropic
        # eps is 2D -> eps is diagonal, possibly anisotropic
        # eps is 3D -> eps is an arbitrary tensor, possibly anisotropic
        if eps is None:
            self.eps = np.ones(self.cells.shape[0])
        elif eps.ndim > 3:
            raise Exception("epsilon needs to be a scalar, a vector of diagonal entries or a tensor")
        elif eps.shape[0] == self.points.shape[0]:
            self.eps = np.mean(eps[self.cells], axis=1)
        elif eps.shape[0] == self.cells.shape[0]:
            self.eps = np.copy(eps)
        else:
            raise Exception("epsilon needs to be defined either per point or per cell")
        if self.eps.ndim == 2: # turning the vector of diagonal entries into a diagonal tensor
            self.eps = np.eye(3).reshape((1,3,3)) * self.eps.reshape((self.cells.shape[0],3,1))
        
        self.eps *= self.eps0

        self.initialize_matrix()


def _initialize_matrix_helper(args):
    coords, cell_vol, variable_pts, dirichlet_vals, eps = args
    s_data = np.zeros(16)
    s_rows = np.zeros(16)
    s_cols = np.zeros(16)
    r_data = np.zeros(16)
    r_rows = np.zeros(16)
    r_cols = np.zeros(16)

    coord_mat = np.column_stack((np.ones(4), coords))
    inv_coord_mat = spla.inv(coord_mat)
    for co1 in range(4):
        for co2 in range(co1,4):
            if variable_pts[co1] == -1 and variable_pts[co2] == -1:
                continue
            if isinstance(eps, np.ndarray):
                assert(eps.shape == (3,3))
                grad_prod = np.dot(inv_coord_mat[1:,co1], eps@inv_coord_mat[1:,co2]) * cell_vol
            else:
                grad_prod = np.dot(inv_coord_mat[1:,co1], inv_coord_mat[1:,co2]) * eps * cell_vol
            if variable_pts[co1] == -1:
                s_data[4*co1 + co2] = -dirichlet_vals[co1] * grad_prod
                s_rows[4*co1 + co2] = variable_pts[co2]
                s_cols[4*co1 + co2] = -1
                continue
            if variable_pts[co2] == -1:
                s_data[4*co1 + co2] = -dirichlet_vals[co2] * grad_prod
                s_rows[4*co1 + co2] = variable_pts[co1]
                s_cols[4*co1 + co2]= -1
                continue
            s_data[4*co1 + co2] = grad_prod
            s_rows[4*co1 + co2] = variable_pts[co1]
            s_cols[4*co1 + co2] = variable_pts[co2]

            if co1 != co2:
                s_data[4*co2 + co1] = grad_prod
                s_rows[4*co2 + co1] = variable_pts[co2]
                s_cols[4*co2 + co1] = variable_pts[co1]
            
            # rho mat
            r_rows[4*co1 + co2] = variable_pts[co1]
            r_cols[4*co1 + co2] = variable_pts[co2]
            if co1 != co2:
                r_data[4*co1 + co2] = cell_vol/20.
                r_data[4*co2 + co1] = cell_vol/20.
                r_rows[4*co2 + co1] = variable_pts[co2]
                r_cols[4*co2 + co1] = variable_pts[co1]
            else:
                r_data[4*co1 + co2] = cell_vol/10.
    
    return (s_data, s_rows, s_cols, r_data, r_rows, r_cols)