import numpy as np
import tensorflow as tf

path_to_dir = "savedModel"
model = tf.keras.models.load_model(path_to_dir)

def my_model(seconde, position):
    """
    un exemple naïf de contrôle avec la première composante le sinus du temps \
    et la deuxième le cosinus de la somme des coordonnées de poisiton
    """
    output = model(np.array([position[0], position[1], seconde]).reshape(1,-1))[0]
    return np.array(output)


def main(seconde, position):
    """
    Votre controle du AG 2.0 au temps t seconde(s) et à la position X_t

    Parameters
    ----------
    seconde: float
        s = 0., 1., .., 9.
    position: arr (2,)
        Position du AG de format (2,) avec coordonnée x = position [0] et coordonnée y = position [1]

    Returns
    -------
    arr (2,)
        Contrôle du AG durant la prochaine seconde.

    Notes
    ----
    La sortie de cette fontion doit être la valeur de votre contrôle :
        un array (de float32 ou float64) et de dimension (2,)

    """
    return my_model(seconde, position)