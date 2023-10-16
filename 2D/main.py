from init import jax
from init import np
from init import tqdm
from init import cmr
from functions import get_rho,get_macro_vels,get_equilibrium_discrete_vels
import init

def main():
        jax.config.update("jax_enable_x64",True)

        #mesh defination
        x=np.arange(init.N_POINTS_X)
        y=np.arange(init.N_POINTS_Y)
        X,Y=np.meshgrid(x,y,indexing="ij")

        #masking the mesh to detect the obstacle
        obj_mask=(np.sqrt((X-init.OBJ_CENTER_X)**2 +(Y-init.OBJ_CENTER_Y)**2)<init.OBJ_RADII_IDX)

        vel_profile = np.zeros((init.N_POINTS_X,init.NPOINTS_Y,2))
        vel_profile = vel_profile.at[:,:,0].set(init.VMAX_IN_X)

        def update(discrete_vels_prev):
                # 1. Apply outflow boundary condition on the right boundary
                discrete_vels_prev=discrete_vels_prev.at[-1,:,init.V_LEFT].set(discrete_vels_prev[-2,:,init.V_LEFT])
        
                # 2. Compute Macroscopic Quantities (density and velocities)
                rho_prev=get_rho(discrete_vels_prev)  
                macro_vels_prev = get_equilibrium_discrete_vels(discrete_vels_prev,rho_prev):

                #3. Apply Inflow Profile by Zou/He Dirichlet Boundary Condition on the left boundary   part i
                macro_vels_prev = macro_vels_prev.at[0, 1:-1, :].set(vel_profile[0, 1:-1, :])
                rho_prev = rho_prev.at[0, :].set((get_rho(discrete_vels_prev[0, :, init.ONLY_Y].T)+
                                                  2 *get_rho(discrete_vels_prev[0, :, init.V_LEFT].T)) / (1 - macro_vels_prev[0, :, 0]))

                # (4) Compute discrete Equilibria velocities
                equilibrium_discrete_vels=get_equilibrium_discrete_vels(macro_vels_prev,rho_prev)

                # (3) Zou He scheme part ii
                discrete_vels_prev = discrete_vels_prev.at[0,:,init.V_RIGHT].set(
                    equilibrium_discrete_vels[0,:,init.V_RIGHT])

                # (5) BGK step
                discrete_vels_post_collision  = discrete_vels_prev-init.OMEGA*(discrete_vels_prev-equilibrium_discrete_vels)

                # (6) bounceback boundary to enforce no slip
                for i in range(init.N_DISCRETE):
                    discrete_vels_post_collision= discrete_vels_post_collision.at[obj_mask,init.LATTICE_IDX[i]].set(
                            discrete_vels_prev[obj_mask,init.LATTICE_OPP_IDX[i]])

                # (7) stream alongside lattice velocities
                discrete_vels_streams=discrete_vels_post_collision
                for i in range(init.N_DISCRETE):
                    discrete_vels_streams=discrete_vels_streams.at[:,:,i].set(np.roll(
                        np.roll(
                            discrete_vels_post_collision[:,:,i],init.LATTICE_VELS[0,i],axis=0
                    ),
                    init.LATTICE_VELS[1,i],axis=1
                    )
                    )
                return discrete_vels_streams
        
        return








if __name__=="__main__":
    main()