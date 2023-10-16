from init import jax
from init import np
from init import tqdm
from init import cmr
from fu 
import init

def main():
    jax.config.update("jax_enable_x64",True)
    kin_viscosity=(
            (init.VMAX_IN_X*init.OBJ_RADII_IDX)
            /
            (init.REYNOLDS_NO)
            )
    omega=1.0 / (3*kin_viscosity + 0.5)
    
    #mesh defination
    x=np.arange(init.N_POINTS_X)
    y=np.arange(init.N_POINTS_Y)
    X,Y=np.meshgrid(x,y,indexing="ij")
    
    #masking the mesh to detect the obstacle
    obj_mask=(np.sqrt((X-init.OBJ_CENTER_X)**2 +(Y-init.OBJ_CENTER_Y)**2)<init.OBJ_RADII_IDX)
    
    vel_profile = np.zeros((init.N_POINTS_X,init.NPOINTS_Y,2))
    vel_profile = vel_profile.at[:,:,0].set(init.VMAX_IN_X)
    
    def update(discrete_vels_prv):
# 1. Apply outflow boundary condition on the right boundary
        discrete_vels_prv=discrete_vels_prv.at[-1,:,init.V_LEFT].set(discrete_vels_prev[-2,:,init.V_LEFT])
     
# 2. Compute Macroscopic Quantities (density and velocities)
        rho_prev=.get_rho(discrete_vels_prev)  
        macro_vels_prev = get_equilibrium_discrete_vels(discrete_vels_prev,rho_prev):

#3. Apply Inflow Profile by Zou/He Dirichlet Boundary Condition on the left boundary   
         macro_vels_prev =\
            macro_vels_prev.at[0, 1:-1, :].set(
                vel_profile[0, 1:-1, :]
            )
        rho_prev = rho_prev.at[0, :].set(
            (
                get_rho(discrete_vels_prev[0, :, ONLY_Y].T)
                +
                2 *
                get_RHO(discrete_vels_prev[0, :, V_LEFT].T)
            ) / (
                1 - macro_vels_prev[0, :, 0]
            )
        )

# (4) Compute discrete Equilibria velocities

if __name__=="__main__":
    main()