import numpy as np

def calculer_kn(L, T, E, I, CL, raideur, N_modes, tol=1e-9, max_iter=50):
    """
    Calcule les valeurs exactes de k_n (et donc gamma_n) pour une corde clamped-clamped.
    - L: longueur de la corde
    - T: tension dans la corde
    - E: module d'Young
    - I: moment d'inertie
    - CL: conditions aux limites (simply_supported / clamped_clamped)
    - N_modes: nombre de modes à calculer
    - raideur: présence de raideur (True/False)

    - T: tension dans la corde
    - E: module d'Young
    - I: moment d'inertie
    - N_modes: nombre de modes à calculer

    On exprime les fonctions hyperboliques à l'aide d'exponentielles pour éviter les problèmes d'explosion numérique.
    """
    if raideur:
        if CL == "clamped_clamped":
            EI = E * I
            k_n = np.zeros(N_modes)
            gamma_n = np.zeros(N_modes)

            for n in range(1, N_modes + 1):
                k = n * np.pi / L # Valeur approchée

                for i in range(max_iter):
                    k_old = k
                    gamma = np.sqrt(k**2 + T/EI)
                    exp_gL = np.exp(-gamma * L)
                    exp_2gL = np.exp(-2 * gamma * L)

                    coth_gL = (1 + exp_2gL) / (1 - exp_2gL)  # coth(gamma * L)
                    inv_sinh_gL = (2 * exp_gL) / (1 - exp_2gL)  # 1/sinh(gamma * L)
                    
                    term_frac = (2 * k * gamma) / (gamma**2 - k**2)
                    term_bracket = coth_gL - (inv_sinh_gL / np.cos(k * L))

                    k = (1 / L) * (n * np.pi + np.arctan(term_frac * term_bracket))

                    if abs(k - k_old) < tol:
                        break

                k_n[n-1] = k
                gamma_n[n-1] = gamma

            return k_n, gamma_n
        
        elif CL == "simply_supported":
            k_n = np.array([(n * np.pi) / L for n in range(1, N_modes + 1)])
            gamma_n = np.zeros(N_modes)  # Pas de raideur pour simply_supported
            return k_n, gamma_n
        
        else:
            raise ValueError("Type de condition aux limites non reconnu. Utilisez 'simply_supported' ou 'clamped_clamped'.")
        
    else: # si pas de raideur, la CL clamped-clamped n'existe pas
        k_n = np.array([(n * np.pi) / L for n in range(1, N_modes + 1)])
        gamma_n = np.zeros(N_modes)  # Pas de raideur pour simply_supported
        return k_n, gamma_n

def calculer_alpha(k, gamma, L):
    """
    Calcule le coefficient de normalisation alpha pour les modes encastrés,
    en évitant l'overflow numérique sur les fonctions hyperboliques.
    """
    exp_gL = np.exp(-gamma * L)
    exp_2gL = np.exp(-2 * gamma * L)

    coth_gL = (1 + exp_2gL) / (1 - exp_2gL)     # coth(gamma * L)
    inv_sinh_gL = (2 * exp_gL) / (1 - exp_2gL)  # 1/sinh(gamma * L)

    # Division du numérateur et dénominateur par sinh(gamma * L)
    num = np.sin(k * L) * inv_sinh_gL - (k / gamma)
    den = np.cos(k * L) * inv_sinh_gL - coth_gL

    alpha = num / den
    return alpha

def dispersion(k_n, L, E, I, rho, T, c, raideur, CL):
    """
    Calcule les omega_ à l'aide de la renlation de dispersion.
    - k_n: nombres d'onde
    - L: longueur de la corde
    - E: module d'Young
    - I: moment d'inertie
    - rho: densité linéique
    - T: tension dans la corde
    - c: vitesse de propagation des ondes transverses
    - CL: conditions aux limites (simply_supported / clamped_clamped)
    """
    if raideur:
        if CL == "simply_supported":
            omega_n = [(n * np.pi * c / L) * np.sqrt(1 + (E * I * np.pi**2 * n**2) / (T * L**2)) for n in range(1, len(k_n) + 1)]
        elif CL == "clamped_clamped":
            omega_n = [np.sqrt((1/rho) * (E * I * k**4 + T * k**2)) for k in k_n]
        else:
            raise ValueError("Type de condition aux limites non reconnu. Utilisez 'simply_supported' ou 'clamped_clamped'.")
    else:
        omega_n = k_n * c

    return omega_n