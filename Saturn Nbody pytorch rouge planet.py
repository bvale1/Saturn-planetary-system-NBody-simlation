import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
import torch.linalg as la
from mpl_toolkits.mplot3d import axes3d
import timeit
import torch
class leapfrog:
    
    def __init__(self, r_0, v_0, r_0_tracers, v_0_tracers, M, R_saturn):
        # All quanities are in SI units
        # Saturn's radius
        self.R_saturn = R_saturn
        # r, v are phase space parameters of massive bodies
        self.r = r_0.detach().clone()
        self.v = v_0.detach().clone()
        # r_tracers, v_tracers are phase space parameters of massless tracers
        self.r_tracers = r_0_tracers.detach().clone()
        self.v_tracers = v_0_tracers.detach().clone()
        # M is an array containing the mass of the massive bodies
        self.M = M.unsqueeze(0).T.unsqueeze(2)
        # Acceleration at time=0 for massive bodies and tracers respectively
        self.a, self.a_tracers = self.acceleration()
        # The overall integration time; iterations
        self.t = 0; self.k = 0
        # Time spent integrating orbits
        self.propogation_time = []
        
    def propogate(self, delta_t):
        # Numerically integrate the orbit of all particles for one timestep
        start = timeit.default_timer()
        # Leapfrog is splits the timestep into two kick and two drift operations
        self.v += self.a * delta_t/2
        self.v_tracers += self.a_tracers * delta_t/2
        self.r += self.v * delta_t
        self.r_tracers += self.v_tracers * delta_t
        
        # Detect collisions and subsequently delete particles
        # Currently only done every 5 iterations to improve performance
        if self.k % 5 == 0:
            collision = torch.logical_not(torch.logical_or(
                                          la.norm(self.r_tracers-self.r[0,:],
                                                  2, axis=1) < self.R_saturn,
                                          la.norm(self.r_tracers-self.r[1,:],
                                                  2, axis=1) < self.R_saturn))
            self.r_tracers = self.r_tracers[collision]
            self.v_tracers = self.v_tracers[collision]
            collision = torch.logical_not(torch.logical_or(
                                          la.norm(self.r-self.r[0,:],
                                                  2, axis=1) < self.R_saturn,
                                          la.norm(self.r-self.r[1,:],
                                                  2, axis=1) < self.R_saturn))
            collision[:2] = True # Assume planets are too massive to be absorbed
            self.r = self.r[collision]
            self.v = self.v[collision]
            self.M = self.M[collision]
        self.k += 1
        # In reality moons and planets that come too close will also be tidally
        # disrupted, causing them to fracture
        
        # Recalculate acceleration and propgate for the second half of the timestep
        self.a, self.a_tracers = self.acceleration()
        self.v += self.a * delta_t/2
        self.v_tracers += self.a_tracers * delta_t/2
        
        self.t += delta_t
        self.propogation_time.append(timeit.default_timer()-start)
        # positions vectors must be returned to the cpu as numpy arrays to be used with matplotlib
        return self.t, self.r.cpu().numpy(), self.r_tracers.cpu().numpy()
        
    def acceleration(self):
        '''
        :param r: x,y,z coordinates of n bodies
        :type r: 2d numpy array of shape (n, 3)
        :param M: array containing masses of n bodies
        :type M: 1d numpy array of length n
        :return: dphi/dx,dphi/dy,dphi/dz of n bodies
        :rtype: 2d numpy array of shape (n, 3)
        '''
        # Compute the acceleration of each of the
        # massive bodies due to one another
        n = self.r.shape[0]
        r_repeat = self.r.unsqueeze(0).repeat(n, 1, 1)
        rij = (r_repeat - torch.transpose(r_repeat, dim0=0, dim1=1))
        grad = rij * (torch.sum(rij**2, dim=2, keepdim=True)**(-3/2))
        # Acceleration of a point particle on itself is infinity due to a
        # singularity, wheras physically it should be zero
        #grad[torch.isnan(grad)] = 0.0
        grad = torch.nan_to_num(grad, 0.0)
        grad = - const.G * torch.sum(self.M * grad, dim=0)

        # Compute the acceleration of each of the 
        # massless tracers due to the massive bodies
        r_repeat = self.r_tracers.unsqueeze(0).repeat(n, 1, 1)
        rij = (r_repeat - torch.transpose(self.r.unsqueeze(0).repeat(
            self.r_tracers.shape[0], 1, 1), dim0=0, dim1=1))
        grad_tracers = rij * (torch.sum(rij**2, dim=2, keepdim=True)**(-3/2))
        grad_tracers = - const.G * torch.sum(self.M * grad_tracers, dim=0)
        #sys.exit(0)
        return grad, grad_tracers

# OLD METHOD
def One_Body_Grad(r, M):
    return -const.G * M * r / (torch.sum(r**2)**(3/2))

# Can be used to rotate the position and velocity vectors about the x axis
# to set the orbital inclination
def Rx(coords, theta):
    return torch.matmul(np.array([[1, 0, 0],
                                  [0, np.cos(theta), -np.sin(theta)],
                                  [0, np.sin(theta), np.cos(theta)]]), coords)

def phase_params(orbit_params, M_saturn):
    n = np.shape(orbit_params)[0]
    
    M = orbit_params[:,0] * 1e20 # kg
    
    # Randomise starting phase
    phi_0 = np.random.random(n) * 2 * np.pi # rads
    
    e = orbit_params[:,4] # dimensionless
    a = orbit_params[:,1] * 1e6 # m
    r_norm = a * (1 - e**2) / (1 + e * np.cos(phi_0))
    v_norm = np.sqrt(const.G * M_saturn * ((2/r_norm) - (1/a)))
    
    r_0 = np.reshape(r_norm, (n, 1)) * np.array([-np.cos(phi_0),
                                                 np.sin(phi_0),
                                                 np.zeros_like(phi_0)]).T
    v_0 = np.reshape(v_norm, (n, 1)) * np.array([np.sin(phi_0),
                                                 np.cos(phi_0),
                                                 np.zeros_like(phi_0)]).T
    
    inc = orbit_params[:,3] * np.pi / 180 # rads
    #for i, coords in enumerate(r_0):
        # r_0 = Rx(coords, inc)

    return r_0, v_0, M
        
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        print(device)
        print(torch.cuda.current_device())
        
        # Saturn fact sheet
        # https://nssdc.gsfc.nasa.gov/planetary/factsheet/saturnfact.html
        M_saturn = 568.32 * 1e24 # kg
        #R_saturn = 60.268 * 1e6 # m
        R_saturn = 30.000 * 1e6 # m (This is half the actual radius of Saturn)
        r_0 = torch.tensor([[0.0, 0.0, 0.0]]).to(device)
        v_0 = torch.tensor([[0.0, 0.0, 0.0]]).to(device)
        M = torch.tensor([M_saturn]).to(device)
        labels = ["Saturn"]
        
        # Saturn's major satellites fact sheet
        # https://nssdc.gsfc.nasa.gov/planetary/factsheet/saturniansatfact.html
        #              M(10^20kg)  a(1e6m)    Torb(days)    inc(deg)    e
        moons = torch.tensor([[0.379,  185.52,	  0.9424218,	  1.53,	    0.0202], # Mimus (SI)
       	                 [1.08,   238.02,	  1.370218,	      0.00,	    0.0045], # Enceladus (SII)
       	                 [6.18,   294.66,	  1.887802, 	  1.86,	    0.0000], # Tethys (SIII)
                         [11.0,   377.40,     2.736915,	      0.02,	    0.0022], # Dione (SIV)
                         [23.1,   527.04,	  4.517500, 	  0.35,	    0.0010], # Rhea (SV)
                         [1345.5, 1221.83,    15.945421,	  0.33,	    0.0292], # Titan (SVI)
                         [0.056,  1481.1,	  21.276609,	  0.43,	    0.1042], # Hyperion (SVII)
                         [18.1,   3561.3,	  79.330183,	  14.72,    0.0283]]) # Iapetus (SVIII)
        
        # Add another saturn sized planet for a gravitational encounter
        r_0 = torch.cat((r_0, torch.tensor([[5.0e8, 2.0e8, 1.0e8]]).to(device)), dim=0)
        v_0 = torch.cat((v_0, torch.tensor([[-16000, 0.0, 0.0]]).to(device)), dim=0)
        M = torch.cat((M, torch.tensor([M_saturn]).to(device)), dim=0)
        labels += ["Rouge Planet"]
        
        # Add Saturn's major satellites
        labels += ["Mimus (SI)", "Enceladus (SII)", "Tethys (SIII)", "Dione (SIV)",
                  "Rhea (SV)", "Titan (SVI)", "Hyperion (SVII)", "Iapetus (SVIII)"]
        phase_params = phase_params(moons, M_saturn)
        
        r_0 = torch.cat((r_0, phase_params[0].to(device)), dim=0)
        v_0 = torch.cat((v_0, phase_params[1].to(device)), dim=0)
        M = torch.cat((M, phase_params[2].to(device)), dim=0)
        
        # Saturnian rings fact sheet
        # https://nssdc.gsfc.nasa.gov/planetary/factsheet/satringfact.html
        # The rings are modelled as n massless tracers which begin on circular
        # orbits evenly distributed between ring_min and ring_max with zero
        # inclination (v_z=0) and each has a random starting phase (ring_phi_0)
        ring_min = 74.500 * 1e6 # m
        ring_max = 140.220 * 1e6 # m
        n_tracers = 10000
        # Generate initial phase space parameters for the tracers
        ring_phi_0 = torch.rand(n_tracers) * 2 * np.pi
        ring_r_norm = torch.reshape(torch.linspace(ring_min, ring_max, n_tracers),
                                    (n_tracers, 1))
        ring_r_0 = (ring_r_norm * torch.cat((-np.cos(ring_phi_0).unsqueeze(1),
                                            np.sin(ring_phi_0).unsqueeze(1),
                                            torch.zeros(n_tracers).unsqueeze(1)), 1)).to(device)
        # Circular orbit solution for ring_v_0
        ring_v_norm = np.sqrt(const.G * M_saturn / ring_r_norm)
        ring_v_0 = (ring_v_norm * torch.cat((np.sin(ring_phi_0).unsqueeze(1), 
                                            np.cos(ring_phi_0).unsqueeze(1),
                                            torch.zeros(n_tracers).unsqueeze(1)), 1)).to(device)
        
        # Initite a leapfrog integrator to model the evolution of the system
        t_max = (0.01) * 365 * 24 * 60 * 60 # years
        t = 0
        delta_t = 100 # seconds
        lf = leapfrog(r_0, v_0, ring_r_0, ring_v_0, M, R_saturn)
        
        # animate orbits
        fig = plt.figure(figsize=(11,10))
        ax = fig.add_subplot(111, projection="3d")
        rings = ax.scatter(ring_r_0.cpu().numpy()[:,0],
                           ring_r_0.cpu().numpy()[:,1], 
                           ring_r_0.cpu().numpy()[:,2], s=1, label="Rings")
        
        for i in range(r_0.shape[0]):
            ax.scatter(r_0.cpu().numpy()[:,0], r_0.cpu().numpy()[:,1],
                       r_0.cpu().numpy()[:,2], s=100, label=labels[i])
        
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.grid(True)
        ax.set_axisbelow(True)
        fig.legend()
        
        while lf.t <= t_max:
            ax.cla()
            t, r, ring_r = lf.propogate(delta_t)
            ax.scatter(ring_r[:,0], ring_r[:,1], ring_r[:,2], s=0.5)
            ax.scatter(r[0,0], r[0,1], r[0,2], s=250)
            ax.scatter(r[1,0], r[1,1], r[1,2], s=250)
            for i in range(2, np.shape(r)[0]):
                ax.scatter(r[i,0], r[i,1], r[i,2], s=50)
        
            ax.set_ylim(-5e8, 5e8)
            ax.set_xlim(-5e8, 5e8)
            ax.set_zlim(-5e8, 5e8)
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.set_zlabel("z (m)")
            plt.draw()
            plt.pause(0.005)

        
        