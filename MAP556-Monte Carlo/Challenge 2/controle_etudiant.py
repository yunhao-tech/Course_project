import numpy as np
import pandas as pd
import argparse
import sys
import mon_controle as MC

"""
Vous ne devez rien modifier dans ce fichier.

Executez ce fichier pour obtenir votre résultat trajectoriel (= performance sur une trajectoire de vent). 
"""


# --------------------------------------------------
#            Initialisation des objets
# --------------------------------------------------
parser = argparse.ArgumentParser(description='Controle')
parser.add_argument('--vent', '-v',
                    dest='filename',
                    metavar="FILE",
                    help="fichier de parametres du vent",
                    default="ventETU.csv")

args = parser.parse_args()

D = 200  # distance en metre au temps T=10 secondes
T = 10  # temps en seconde
DELTA_T = 0.1  # pas de temps pour discrétiser en seconde
G = np.array([0., -4])  # force de gravité en m/s^2
LAMBDA = 0.01  # coefficient de resitance de l'air en kg/s

POSITION_INIT_x = 0.  # coordonnée x de la position initiale
POSITION_INIT_y = 0.  # coordonnée y de la position initiale

VITESSE_INIT_x = D / T  # coordonnée x en m/s de la vitesse initiale
VITESSE_INIT_y = (T/2) * np.linalg.norm(G)  # coordonnée y en m/s de la vitesse initiale
VITESSE_INIT = np.array([VITESSE_INIT_x, VITESSE_INIT_y])  # vitesse initiale sous forme d'array

position_actuelle = np.array([POSITION_INIT_x, POSITION_INIT_y])  # position initiale
cout_controle_trajectoriel = 0.  # initialisation du cout de controle

df_vent = pd.read_csv(args.filename)  # lecture du fichier ventETU.csv
# --------------------------------------------------


# --------------------------------------------------
#                   Fonctions
# --------------------------------------------------
def dynamique_position_sur_un_pas_de_temps(position_actuelle, temps, vent_actuel_array, controle_actuel):
    """
    Dynamique de la position discrétisée au pas DELTA_T seconde entre deux pas de temps t_i et t_i+1

    Parameters
    ----------
    position_actuelle: arr (2,)
    temps: float
    vent_actuel_array: arr (2, )
    controle_actuel: arr (2,)

    Returns
    -------
    arr (2,)
        Position au temps t_i+1 = t_i + DELTA_T
    """

    terme1 = VITESSE_INIT * DELTA_T
    terme2 = G/2 * ((temps + DELTA_T)**2 - temps**2)
    terme3 = LAMBDA * position_actuelle * DELTA_T
    terme4 = vent_actuel_array * DELTA_T
    terme5 = controle_actuel * DELTA_T
    return position_actuelle + terme1 + terme2 - terme3 + terme4 + terme5


def perte_terminale(position):
    """
    Fonction de coût terminale

    Parameters
    ----------
    position: arr (2,)

    Returns
    -------
    float
    """
    u1 = ((position[0] - D) - position[1]) / np.sqrt(2)
    u2 = ((position[0] - D) + position[1]) / np.sqrt(2)
    u3 = position[0] + position[1] - (D-15)
    return (u1 + u1 * (u1>0))**2 + u2**2 + (u3*(u3<0))**2


def ecriture_resultat(resultat):
    """
    Ecriture d'un resultat dans un fichier text au nom de "mon_resultat.txt"

    Parameters
    ----------
    resultat: float

    Returns
    -------
    None
    """
    with open("mon_resultat.txt", "w") as file_result:
        file_result.write("{:.6f}".format(resultat))
    return
# --------------------------------------------------

# Jouons le contrôle de l'oiseau avec un scénario de vent
for seconde in range(T):
# ==========================  Votre fonction ============================
    controle_actuel = MC.main(seconde=seconde, position=position_actuelle)
# ========================================================================

# =================== Verification de la sortie ======================
    try:
        assert controle_actuel.shape == (2,)
    except AssertionError as e:
        print("Le format de votre sortie n'est pas conforme.")
        ecriture_resultat(np.nan)
        sys.exit()

    try:
        assert type(controle_actuel) is np.ndarray
    except AssertionError as e:
        print("Le type de votre sortie n'est pas conforme.")
        ecriture_resultat(np.nan)
        sys.exit()
# ========================================================================

    timesteps = np.arange(seconde, seconde + 1, DELTA_T)  # fenetre de discretisation entre deux secondes consécutives
    for timestep in timesteps:
        vent_actuel = df_vent[df_vent["date"] == timestep.round(2)]
        t_i = vent_actuel["date"].to_numpy()
        vent_actuel_array = vent_actuel[["vent_x", "vent_y"]].to_numpy().ravel()

        position_actuelle = dynamique_position_sur_un_pas_de_temps(position_actuelle=position_actuelle,
                                                                   temps=t_i,
                                                                   vent_actuel_array=vent_actuel_array,
                                                                   controle_actuel=controle_actuel)

    cout_controle_trajectoriel += np.linalg.norm(controle_actuel) ** 2

cout_controle_trajectoriel += perte_terminale(position_actuelle)

ecriture_resultat(cout_controle_trajectoriel)


