import json
from pathlib import Path

import numpy as np
from utils import calculer_alpha, calculer_kn, dispersion
from scipy.integrate import trapezoid
from scipy.io.wavfile import write

def phi(x, k, gamma, alpha, CL):
    """
    Évalue la déformée spatiale en x.
    - x: position spatiale
    - k: nombre d'onde k_n (solutions imaginaires pures de l'équation caractéristique)
    - gamma: solutions réelles de l'équation caractéristique
    - alpha: coefficient de normalisation
    - CL: type de condition aux limites (simply_supported / clamped_clamped)
    """
    if CL == "simply_supported":
        return np.sin(k * x)
    elif CL == "clamped_clamped":
        return np.sin(k * x) - (k / gamma) * np.sinh(gamma * x) - alpha * (np.cos(k * x) - np.cosh(gamma * x))
    else:
        raise ValueError("Type de condition aux limites non reconnu. Utilisez 'simply_supported' ou 'clamped_clamped'.")

def synthese_corde(L, T, E, I, rho, CL, raideur, amortissement, beta, eta, x_0, l_0, h_0, v_0, tau_rel, x_lst, duree, fs, N_modes):
    """
    Corde 1D
    Pas d'ondes de torsion / de flexion -> seulement des ondes transverses
    Pour l'instant, corde homogène (en tension et section)
    Pas encore d'amortissement, seulement raideur

    Équation vérifiée par y(x, t):
    d²y/dt² - c² * d²y/dx² + EI/rho * d⁴y/dx⁴ = 0 où I = pi * r⁴ / 4
    """
    t = np.linspace(0, duree, int(duree*fs), endpoint=False) # vecteur temps

    x_grille = np.linspace(0, L, 1000) # grille spatiale pour l'évaluation de la déformée

    y_0 = np.piecewise(
        x_grille,
        [x_grille < x_0, x_grille <= x_0 + l_0],
        [lambda x: (h_0 / x_0) * x,
            h_0,
            lambda x: (h_0 / (L - x_0 - l_0)) * (L - x)
        ],
    ) # profil initial (position)

    v_0 = np.piecewise(
        x_grille,
        [x_grille < x_0, x_grille <= x_0 + l_0],
        [lambda x: (v_0 / x_0) * x,
            v_0,
            lambda x: (v_0 / (L - x_0 - l_0)) * (L - x)
        ],
    ) # profil initial (vitesse)

    signal_total = np.zeros_like(t)

    # Calcul des k_n (et gamma_n) pour les modes
    k_n, gamma_n = calculer_kn(L, T, E, I, CL, raideur, N_modes)

    c = np.sqrt(T / rho) # vitesse de propagation des ondes transverses

    omega_0n = dispersion(k_n, L, E, I, rho, T, c, raideur, CL) # pulsations à vide des modes

    for n in range(1, N_modes+1):
        k = k_n[n-1]
        gamma = gamma_n[n-1] # 0 si CL == "simply_supported"

        print(f"Mode {n}: k_n = {k:.4f}, f_n = {omega_0n[n-1] / (2 * np.pi):.4f}")

        if CL == "clamped_clamped":
            alpha = calculer_alpha(k, gamma, L)
        else:
            alpha = 0.0

        # Déformée spatiale pour le mode n
        forme_sur_grille = phi(x_grille, k, gamma, alpha, CL)
        N_n = np.trapezoid(forme_sur_grille**2, x_grille)

        # Projection des conditions initiales sur le mode n
        A_n = np.trapezoid(y_0 * forme_sur_grille, x_grille) / N_n # position
        B_n = np.trapezoid(v_0 * forme_sur_grille, x_grille) / N_n # vitesse
        phi_lst = phi(x_lst, k, gamma, alpha, CL)

        omega_0 = omega_0n[n-1]

        if amortissement:
            sigma_n = 0.5 * (2 * beta + eta * k**2)
            omega_d = np.sqrt(omega_0**2 - sigma_n**2)
            coef_n = (B_n + sigma_n * A_n) / omega_d

            q_n = np.exp(-sigma_n * t) * (A_n * np.cos(omega_d * t) + coef_n * np.sin(omega_d * t)) * (1.0 - np.exp(-t / tau_rel))

            signal_total += phi_lst * q_n
        else:
            coef_n = B_n / omega_0
            q_n = A_n * np.cos(omega_0 * t) + coef_n * np.sin(omega_0 * t)
            signal_total += phi_lst * q_n

    signal_audio = signal_total / np.max(np.abs(signal_total))

    return t, signal_audio

if __name__ == "__main__":
    config_path = Path(__file__).parent / "config.json"
    with config_path.open(encoding="utf-8") as f:
        config = json.load(f)

    geometry = config["geometry"]
    material = config["material"]
    model = config["model"]
    playing = config["playing"]
    synthesis = config["synthesis"]

    CL = model["CL"]
    if CL not in ["simply_supported", "clamped_clamped"]:
        raise ValueError("Type de condition aux limites non reconnu. Utilisez 'simply_supported' ou 'clamped_clamped'.")

    raideur = model["stiffness"]
    amortissement = model["damping"]

    L = geometry["length"]
    T = material["tension"]
    E = material["young_mod"]
    r = geometry["radius"]
    I = np.pi * r**4 / 4
    rho = material["density"]

    beta = material["beta"]
    eta = material["eta"]

    if not amortissement:
        beta = 0.0
        eta = 0.0

    ponctuel = playing["ponctual"]

    x_0 = playing["x_0"] * L
    h_0 = playing["h_0"]
    l_0 = playing["l_0"]
    v_0 = playing["v_0"]
    tau_rel = playing["tau_rel"]
    x_lst = synthesis["x_lst"] * L

    if ponctuel:
        l_0 = 0.0

    duree = synthesis["duration"]
    fs = synthesis["fs"]
    N_modes = synthesis["N_modes"]

    print("### PARAMÈTRES DE SYNTHÈSE ###")
    print(f"Longueur de la corde: {L} m")
    print(f"Tension: {T} N")
    print(f"Module de Young: {E} Pa")
    print(f"Rayon de la corde: {r} m")
    print(f"Densité linéique: {rho} kg/m")
    print(f"Position de la frappe: {x_0} m")
    print(f"Hauteur de la frappe: {h_0} m")
    print(f"Durée de la synthèse: {duree} s")
    print(f"Taux d'échantillonnage: {fs} Hz")
    print(f"Nombre de modes: {N_modes}")

    temps, audio = synthese_corde(L, T, E, I, rho, CL, raideur, amortissement, beta, eta, x_0, l_0, h_0, v_0, tau_rel, x_lst, duree, fs, N_modes)

    output_directory = Path("output")
    output_directory.mkdir(exist_ok=True)
    raideur_str = "_raid_" if raideur else ""
    amort_str = "_amort_" if amortissement else ""
    ponctuel_str = "_delta_" if ponctuel else "_long_"
    CL_str = f"_{CL}_"
    write(output_directory / f"corde_{N_modes}{amort_str}{raideur_str}{ponctuel_str}{CL_str}.wav", fs, audio.astype(np.float32))