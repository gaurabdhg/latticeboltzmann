from init import np,cmr,plt
import init


def get_rho(discrete_vels):
    # discrete velocity    : (N_x, N_y, 9)
    # Density:ρ = ∑ᵢ fᵢ
    rho = np.sum(discrete_vels, axis=-1)
    return rho


def get_macro_vels(discrete_vels, rho):
    # Velocities: u = 1/ρ ∑ᵢ fᵢ cᵢ
    macro_vels = (
        np.einsum(
            "NMQ,dQ->NMd",
            discrete_vels,
            init.LATTICE_VELS,
        )
        / rho[..., np.newaxis]
    )
    return macro_vels


def get_equilibrium_discrete_vels(macro_vels, rho):
    # Equilibrium:fᵢᵉ = ρ Wᵢ (1 + 3 cᵢ ⋅ u + 9/2 (cᵢ ⋅ u)² − 3/2 ||u||₂²)
    projections = np.einsum("dQ,NMd->NMQ", init.LATTICE_VELS, macro_vels)

    magnitude = np.linalg.norm(macro_vels, axis=-1, ord=2)
    equi_discrete_vels = (
        rho[..., np.newaxis]
        * init.LATTICE_WEIGHTS[np.newaxis, np.newaxis, :]
        * (
            1
            + 3 * projections
            + 4.5 * projections**2
            - 1.5 * magnitude[..., np.newaxis] ** 2
        )
    )

    return equi_discrete_vels


def plots(X,Y,vel_magnitude,curl):
# Velocity Magnitude Contour Plot in the top
    plt.subplot(211)
    plt.contourf(
        X,
        Y,
        vel_magnitude,
        levels=50,
        cmap=cmr.amber,
    )
    plt.colorbar().set_label("Velocity Magnitude")
    plt.gca().add_patch(plt.Circle(
        (init.OBJ_CENTER_X,init.OBJ_CENTER_Y),
        init.OBJ_RADII_IDX,
        color="darkgreen",
    ))
    # Vorticity Magnitude Contour PLot in the bottom
    plt.subplot(212)
    plt.contourf(
        X,
        Y, 
        curl,
        levels=50,
        cmap=cmr.redshift,
        vmin=-0.02,
        vmax= 0.02,
    )
    plt.colorbar().set_label("Vorticity Magnitude")
    plt.gca().add_patch(plt.Circle(
        (init.OBJ_CENTER_X,init.OBJ_CENTER_Y),
        init.OBJ_RADII_IDX,
        color="darkgreen",
    ))
    plt.draw()
    plt.pause(0.0001)
    #plt.clf()
    return