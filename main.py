import streamlit as st
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Imports des scripts de calcul (supposés présents dans le même dossier)
import synthese
from utils import calculer_kn, calculer_alpha

# ---------------------------------------------------------
# Fonction de tracé des modes
# ---------------------------------------------------------
def visualiser_modes_corde(L, T, E, I, CL, raideur):
    """
    Calcule et trace les 12 premières déformées modales de la corde.
    Retourne la figure matplotlib pour l'afficher dans Streamlit.
    """
    N_modes_plot = 12
    # On récupère les k_n et gamma_n pour les 12 premiers modes
    k_n, gamma_n = calculer_kn(L, T, E, I, CL, raideur, N_modes_plot)
    
    x = np.linspace(0, L, 1000)
    fig, axes = plt.subplots(3, 4, figsize=(14, 9), sharex=True)
    
    for index, ax in enumerate(axes.flat, start=1):
        if index > len(k_n):
            break
            
        k = k_n[index-1]
        gamma = gamma_n[index-1]
        
        if CL == "clamped_clamped":
            alpha = calculer_alpha(k, gamma, L)
        else:
            alpha = 0.0
            
        # Appel à la fonction phi du script synthese.py
        forme = synthese.phi(x, k, gamma, alpha, CL)
        
        # Normalisation pour l'affichage
        maximum = np.max(np.abs(forme))
        if maximum > 0:
            forme = forme / maximum
            
        ax.plot(x, forme, color="steelblue")
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_title(f"Mode {index}")
        ax.set_ylabel("Amplitude norm.")
        ax.grid(True, alpha=0.3)
        
    for ax in axes[-1]:
        ax.set_xlabel("Position x (m)")
        
    fig.suptitle(f"Formes spatiales des 12 premiers modes ({CL})")
    fig.tight_layout()
    
    return fig

# ---------------------------------------------------------
# Interface Streamlit
# ---------------------------------------------------------
st.markdown("""
<style>
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.25rem !important; }
    [data-testid="stSidebar"] .block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

st.title("Synthèse Corde 1D - GUI")

with st.sidebar:
    st.header("Paramètres")
    
    st.subheader("Modèle")
    amortissement = st.checkbox("Amortissement", value=True, key="amortissement")
    raideur = st.checkbox("Raideur de la corde", value=False, key="raideur")
    
    # CL forcée à simply_supported
    st.selectbox("Conditions aux limites (CL)", options=["simply_supported"], disabled=True)
    CL = "simply_supported"

    st.subheader("Géométrie")
    L = st.number_input("Longueur L (m)", value=0.65, format="%.3f", step=0.01, key="L")
    r_mm = st.number_input("Rayon r (mm)", value=0.55, format="%.3f", step=0.01, key="r_mm")
    r = r_mm / 1000.0

    st.subheader("Matériau")
    T = st.number_input("Tension T (N)", value=100.0, step=10.0, key="T")
    rho = st.number_input("Densité linéique ρ (kg/m)", value=0.006, format="%.4f", step=0.001, key="rho")
    E = st.number_input("Module de Young E (Pa)", value=3e9, format="%e", step=1e8, key="E")
    beta = st.number_input("Frottement visqueux β", value=1.4, step=0.1, disabled=not amortissement, key="beta")
    eta = st.number_input("Viscoélasticité η", value=5e-4, format="%e", step=1e-5, disabled=not amortissement, key="eta")

    st.subheader("Excitation")
    ponctuel = st.checkbox("Excitation ponctuelle", value=False, key="ponctuel")
    l_0 = st.number_input("Largeur d'excitation l_0 (m)", value=0.01, format="%.3f", step=0.005, disabled=ponctuel, key="l_0")
    h_0 = st.number_input("Hauteur initiale h_0 (m)", value=0.0025, format="%.4f", step=0.0005, key="h_0")
    v_0 = st.number_input("Vitesse initiale v_0 (m/s)", value=0.5, step=0.1, key="v_0")
    tau_rel = st.number_input("Temps de relâchement τ_rel (s)", value=0.005, format="%.4f", step=0.001, key="tau_rel")

    st.subheader("Synthèse")
    duree = st.number_input("Durée de la synthèse (s)", value=5.0, step=0.1, key="duree")
    fs = st.number_input("Fréquence d'échantillonnage (Hz)", value=44100, step=1000, key="fs")
    N_modes = st.number_input("Nombre de modes", value=128, min_value=1, max_value=500, step=1, key="N_modes")


# Valeurs calculées
I = np.pi * r**4 / 4
l_0_val = 0.0 if ponctuel else l_0

# Initialisation des positions relatives sur la corde (ratio ∈ [0, 1])
if "x_0_ratio" not in st.session_state:
    st.session_state.x_0_ratio = 0.73
if "x_lst_ratio" not in st.session_state:
    st.session_state.x_lst_ratio = 0.73
if "last_click" not in st.session_state:
    st.session_state.last_click = None

st.header("Positions sur la corde")

click_type = st.radio(
    "Sélectionner le paramètre à placer :",
    options=["Excitation", "Écoute"],
    horizontal=True
)

# ---- DESSIN DE LA CORDE (1D) ----
IMG_WIDTH = 700
IMG_HEIGHT = 100
img = Image.new("RGB", (IMG_WIDTH, IMG_HEIGHT), "#787171")
draw = ImageDraw.Draw(img)

# Dessin de la corde (ligne horizontale centrale)
corde_y = IMG_HEIGHT // 2
padding = 20
corde_longueur_px = IMG_WIDTH - 2 * padding
draw.line([(padding, corde_y), (IMG_WIDTH - padding, corde_y)], fill="white", width=4)
# Supports de la corde
draw.rectangle([(padding-5, corde_y-10), (padding, corde_y+10)], fill="black")
draw.rectangle([(IMG_WIDTH-padding, corde_y-10), (IMG_WIDTH-padding+5, corde_y+10)], fill="black")

# Placer le point d'excitation (bleu)
if st.session_state.x_0_ratio is not None:
    x_px = padding + int(st.session_state.x_0_ratio * corde_longueur_px)
    draw.ellipse([(x_px-6, corde_y-6), (x_px+6, corde_y+6)], fill="blue", outline="black")
    if not ponctuel:
        # Afficher la largeur de l'excitation
        l_0_px = int((l_0_val / L) * corde_longueur_px)
        draw.line([(x_px, corde_y+15), (x_px + l_0_px, corde_y+15)], fill="blue", width=3)

# Placer le point d'écoute (rouge)
if st.session_state.x_lst_ratio is not None:
    x_px = padding + int(st.session_state.x_lst_ratio * corde_longueur_px)
    draw.ellipse([(x_px-6, corde_y-6), (x_px+6, corde_y+6)], fill="red", outline="black")

st.write("Cliquez sur la ligne blanche pour sélectionner la position de l'impact (bleu) ou de l'écoute (rouge).")
clic = streamlit_image_coordinates(img, key="corde_click")

# Gestion du clic
if clic is not None and clic != st.session_state.last_click:
    st.session_state.last_click = clic
    x_clic = clic['x']
    
    # Contraindre le clic dans les limites de la corde
    x_clic = max(padding, min(x_clic, IMG_WIDTH - padding))
    ratio = (x_clic - padding) / corde_longueur_px
    
    if click_type == "Excitation":
        st.session_state.x_0_ratio = ratio
    elif click_type == "Écoute":
        st.session_state.x_lst_ratio = ratio

    st.rerun()

st.divider()
col1, col2 = st.columns(2)

x_0_phys = st.session_state.x_0_ratio * L
x_lst_phys = st.session_state.x_lst_ratio * L

with col1:
    st.info(f"**Excitation :** x = {x_0_phys:.3f} m (ratio = {st.session_state.x_0_ratio:.2f})")
with col2:
    st.error(f"**Écoute :** x = {x_lst_phys:.3f} m (ratio = {st.session_state.x_lst_ratio:.2f})")

st.header("Résultats")

if st.button("SIMULATION", type="primary"):
    
    with st.spinner("Synthèse en cours (cela peut prendre quelques secondes)..."):
        
        beta_val = beta if amortissement else 0.0
        eta_val = eta if amortissement else 0.0

        # Lancement de la synthèse
        try:
            temps, audio = synthese.synthese_corde(
                L, T, E, I, rho, CL, raideur, amortissement, 
                beta_val, eta_val, x_0_phys, l_0_val, h_0, v_0, 
                tau_rel, x_lst_phys, duree, fs, N_modes, x_lst=x_lst_phys 
            )
        except TypeError:
            # Fallback si votre fonction n'accepte pas x_lst en argument 
            synthese.x_lst = x_lst_phys
            temps, audio = synthese.synthese_corde(
                L, T, E, I, rho, CL, raideur, amortissement, 
                beta_val, eta_val, x_0_phys, l_0_val, h_0, v_0, 
                tau_rel, x_lst_phys, duree, fs, N_modes
            )
        
        # Normalisation
        max_amp = np.max(np.abs(audio))
        audio_norm = audio / max_amp if max_amp > 0 else audio
        audio_norm = np.float32(audio_norm)

    st.success("Terminé.")

    # Lecteur audio
    st.subheader("Audio")
    st.audio(audio_norm, sample_rate=fs)

    # Forme d'onde & spectrogramme
    st.subheader("DSP")
    
    col_onde, col_spec = st.columns(2)
    
    with col_onde:
        st.markdown("**Forme d'onde**")
        fig_wave, ax_wave = plt.subplots(figsize=(6, 4))
        ax_wave.plot(temps, audio, color='#1f77b4', linewidth=0.8)
        ax_wave.set_xlabel("Temps (s)")
        ax_wave.set_ylabel("Amplitude")
        ax_wave.grid(True, alpha=0.3)
        st.pyplot(fig_wave)
        
    with col_spec:
        st.markdown("**Spectrogramme**")
        fig_spec, ax_spec = plt.subplots(figsize=(6, 4))
        Pxx, freqs, bins, im = ax_spec.specgram(audio, Fs=fs, NFFT=2048, cmap='magma')
        ax_spec.set_xlabel("Temps (s)")
        ax_spec.set_ylabel("Fréquence (Hz)")
        fig_spec.colorbar(im, ax=ax_spec, label="Intensité (dB)")
        st.pyplot(fig_spec)

    # --- AJOUT : Affichage des déformées modales ---
    st.subheader("12 premières déformées modales")
    with st.spinner("Génération des déformées..."):
        try:
            fig_modes = visualiser_modes_corde(L, T, E, I, CL, raideur)
            st.pyplot(fig_modes)
        except Exception as e:
            st.error(f"Une erreur est survenue lors de l'affichage des modes : {e}")