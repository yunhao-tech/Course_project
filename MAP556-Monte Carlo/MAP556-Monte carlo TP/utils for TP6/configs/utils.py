import os
import glob
import seaborn as sns
import matplotlib.pyplot as plt
from IPython import display
from IPython.display import Image
import imageio
import numpy as np
from scipy import stats
import tensorflow as tf
from matplotlib.gridspec import GridSpec

def make_directory(path, name):
    try:
        os.mkdir(path + name)
    except FileNotFoundError:
        print("Creation of the directory {} failed - path does not exist".format(name))
    except FileExistsError:
        print("Directory already exists")
    else:
        print("Successfully created the directory {}".format(name))
    return


# Gaussian

def viz_gaussian_train(uniform, gaussian_data):
    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    uniform_idx_sorted = uniform.argsort(axis=0)
    uniform_sorted, data_sorted = uniform[uniform_idx_sorted].ravel(), gaussian_data[uniform_idx_sorted].ravel()

    ax1.plot(uniform_sorted, data_sorted)
    ax1.set_title("Fonction quantile $\Phi^{-1}(u)$")

    sns.distplot(gaussian_data, ax=ax2, kde=True, label="$\mu$={:.2f}, $\sigma^2$={:.2f}".format(gaussian_data.mean(),                                                                         gaussian_data.std()))
    ax2.set_title("Distribution $p_X$")

    ax2.legend()
    plt.tight_layout()
    sns.despine()
    return

def viz_gaussian_gan(uniform, gaussian_data, generator, discriminator,  list_loss_G, list_loss_D, epoch):
    dict_arguments = locals()  # dictionnaire de tous les arguments de la fonciton {"nom_argument": variable}
    dict_viz_functions = {"unidim": viz_gaussian_gan_unidim, "multidim": viz_gaussian_gan_multidim}
    noise_dim = uniform.shape[1]

    if noise_dim == 1:
        dict_viz_functions["unidim"](**dict_arguments)
    else:
        dict_viz_functions["multidim"](**dict_arguments)
    return

def viz_gaussian_gan_unidim(uniform, gaussian_data, generator, discriminator,  list_loss_G, list_loss_D, epoch):
    """
    Visualisation de statistiques du GAN
    Parameters
    ----------
    uniform: données input
    gaussian_data: données réelles ou simulées
    generator: modèle
    discriminator: modèle
    epoch: visualisation à une epoch précise
    -------
    """
    fig = plt.figure(figsize=(15, 15))
    gs = GridSpec(3, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, :])

    # sort data with uniform
    uniform_sorted=tf.sort(uniform, axis=0)
    generated_data = generator(uniform_sorted)
    

    # Function inverse cdf et generateur
    data_inverse_cdf = stats.norm.ppf(uniform_sorted).astype(np.float32)
    ax1.plot(uniform_sorted, data_inverse_cdf, label="réelle")
    ax1.plot(uniform_sorted, generated_data, label="GAN")
    ax1.set_title("Fonction quantile et generateur GAN")
    ax1.legend()

    # PDF
    sns.histplot(gaussian_data, stat='density',ax=ax2, kde=True,  palette=["#1f77b4"],
        label="réelle ($\mu$={:.2f}, $\sigma$={:.2f})".format(
        gaussian_data.mean(), gaussian_data.std()))
    sns.histplot(generated_data.numpy(), stat='density',ax=ax2, kde=True, palette=["orange"],
        label="GAN ($\mu$={:.2f}, $\sigma$={:.2f})".format(
        generated_data.numpy().mean(), generated_data.numpy().std()))
    ax2.set_title("Distribution")
    ax2.legend(loc="upper right")
    ax2.set_ylim(0., 0.41)
    
    # QQ-plot
    data_sorted = np.sort(gaussian_data, axis=0)
    generated_data_sorted = tf.sort(generated_data, axis=0)
    ax3.scatter(data_sorted, generated_data_sorted)
    ax3.set_xlabel("données réelles")
    ax3.set_ylabel("données simulées")
    ax3.set_ylim(-5., 5.)
    ax3.set_title("QQ-plot")

    # Discriminator Score
    scores = tf.nn.sigmoid(discriminator(data_sorted))
    ax4.plot(data_sorted, scores)
    ax4.fill_between(data_sorted.ravel(), 0, scores.numpy().ravel(), alpha=.3)
    ax4.axhline(y=0.5, color="k")
    ax4.set_xlabel("gaussiennes réelles")
    ax4.set_ylabel("score du Discriminateur")
    ax4.set_title("Discriminateur")
    ax4.set_ylim(0, 1.01)

    # # Loss functions
    epochs = np.arange(len(list_loss_G))
    if epoch == 0:
        ax5.scatter(epochs * 10, list_loss_G, label="Loss Générateur")
        ax5.scatter(epochs * 10, list_loss_D, label="Loss Discriminateur")
    else:
        ax5.plot(epochs * 10, list_loss_G, label="Loss Générateur")
        ax5.plot(epochs * 10, list_loss_D, label="Loss Discriminateur")
    ax5.set_xlabel("epochs")
    ax5.set_title("Fonctions de coût")
    ax5.legend()

    fig.suptitle("Entraînement du GAN (epoch={})".format(epoch), size=14, y=1.)
    plt.tight_layout()
    sns.despine()
    return

def viz_gaussian_gan_multidim(uniform, gaussian_data, generator, discriminator, list_loss_G, list_loss_D, epoch):
    fig = plt.figure(figsize=(15, 10))

    gs = GridSpec(2, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    generated_data = generator(uniform)

    # PDF
    sns.histplot(gaussian_data, stat="density", ax=ax1, kde=True, palette=["#1f77b4"], 
        label="réelle ($\mu$={:.2f}, $\sigma$={:.2f})".format(
        gaussian_data.mean(), gaussian_data.std()))
    sns.histplot(generated_data.numpy(), stat="density", ax=ax1, kde=True, palette=["orange"], 
        label="GAN ($\mu$={:.2f}, $\sigma$={:.2f})".format(
        generated_data.numpy().mean(), generated_data.numpy().std()))
    ax1.set_title("Distribution")
    ax1.legend(loc="upper right")
    ax1.set_ylim(0, 0.41)


    # Discriminator Score
    data_sorted = np.sort(gaussian_data, axis=0)
    scores = tf.nn.sigmoid(discriminator(data_sorted))
    ax2.plot(data_sorted, scores)
    ax2.fill_between(data_sorted.ravel(), 0, scores.numpy().ravel(), alpha=.3)
    ax2.set_xlabel("gaussiennes réelles")
    ax2.set_ylabel("score du Discriminateur")
    ax2.set_title("Discriminateur")
    ax2.set_ylim(0, 1.01)

    plt.tight_layout()
    sns.despine()
    fig.suptitle("Entraînement du GAN(epoch={})".format(epoch), size=14, y=1.)

    # # Loss functions
    epochs = np.arange(len(list_loss_G))
    if epoch == 0:
        ax3.scatter(epochs * 10, list_loss_G, label="Loss Générateur")
        ax3.scatter(epochs * 10, list_loss_D, label="Loss Discriminateur")
    else:
        ax3.plot(epochs * 10, list_loss_G, label="Loss Générateur")
        ax3.plot(epochs * 10, list_loss_D, label="Loss Discriminateur")
    ax3.set_xlabel("epochs")
    ax3.set_title("Fonctions de coût")
    ax3.legend()

    fig.suptitle("Entraînement du GAN (epoch={})".format(epoch), size=14, y=1.)
    plt.tight_layout()
    sns.despine()
    return

def save_GIF(path_dir, n_data, latent_dim, batch_size, epochs):
    """
    créeer un GIF à partir d'image
    Parameters
    ----------
    path_dir: chemin d'enregistrement
    name: nom du GAN "gaussian" ou "BB"
    -------
    """

    anim_file = os.path.join(path_dir, "imgs", "gaussian", "{}_gan_N{}-Ldim{}-epochs{}-bs{}.gif".format("gaussian", n_data, latent_dim,
                                                                        epochs, batch_size))

    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob(os.path.join(path_dir, "imgs", "gaussian", "image_gaussian_N{}-Ldim{}-bs{}*_on_{}.png".format(
                                                                                      n_data, latent_dim, batch_size, epochs)))
        filenames = sorted(filenames)
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        # image = imageio.imread(filename)
        #         # writer.append_data(image)

    # suppression des images .png après création du GIF
    # [os.remove(file) for file in os.listdir(os.path.join(path_dir, "imgs", "gaussian")) if file.endswith('.png')]
    return

def save_data_gaussian(dataset, path_dir):
    """
    dataset: données à savegarder
    """
    n_data = dataset.shape[0]
    data_file_name = os.path.join(path_dir, "data", "gaussian_train-n{}.npy".format(n_data))
    np.save(data_file_name, dataset)
    return

def load_data_gaussian(n_data, path_dir):
    """
    n_data: nombre de données
    path_dir: chemin d'accès
    """
    data_file_name = os.path.join(path_dir, "data", "gaussian_train-n{}.npy".format(n_data))
    return np.load(data_file_name)


def load_model_gaussian(n_data, latent_dim, epochs, batch_size, path_dir, type_model="generator", final=False):
    """type_model:
        str: "generator" or "discriminator" """
    if not final:
        model_file_name = os.path.join(path_dir, "models", "gaussian", "{}-N{}-Ldim{}-epoch{}-bs{}.h5".format(type_model,
                                                                            n_data, latent_dim, epochs, batch_size))
    else:
        model_file_name = os.path.join(path_dir, "models", "gaussian", "final", "{}-N{}-Ldim{}-epoch{}-bs{}.h5".format(type_model,
                                                                            n_data, latent_dim, epochs, batch_size))
    return tf.keras.models.load_model(model_file_name)


def read_GIF_gaussian(n_data, latent_dim, epochs, batch_size, path_dir, final=False):
    if not final:
        img_file_name = os.path.join(path_dir, "imgs", "gaussian", "gaussian_gan_N{}-Ldim{}-epochs{}-bs{}.gif".format(
                                                                            n_data, latent_dim, epochs, batch_size ))
    else:
        img_file_name = os.path.join(path_dir, "imgs", "gaussian", "final", "gaussian_gan_N{}-Ldim{}-epochs{}-bs{}.gif".format(
                                                                            n_data, latent_dim, epochs, batch_size ))

    with open(img_file_name, 'rb') as file_imgs:
        display.display(Image(data=file_imgs.read(), format='png'))
    return 


def read_img_gaussian(n_data, latent_dim, epochs, batch_size, path_dir, final=False):
    if not final:
        img_file_name = os.path.join(path_dir, "imgs", "gaussian", "image_gaussian_N{}-Ldim{}"
                                               "-bs{}_at_epoch_{}_on_{}.png".format(
                                                                            n_data, latent_dim, batch_size, epochs ,epochs))
    else:
        img_file_name = os.path.join(path_dir, "imgs", "gaussian", "final", "image_gaussian_N{}-Ldim{}"
                                               "-bs{}_at_epoch_{}_on_{}.png".format(
                                                                            n_data, latent_dim, batch_size, epochs,epochs ))
    return Image(filename=img_file_name)
