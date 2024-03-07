import streamlit as st 
import streamlit.components.v1 as components
from PIL import Image
import os 
import base64
import run_ZSSR
from loguru import logger
from datetime import datetime
import glob
import cv2
import numpy as np 

def afficher_pdf(pdf_file):
        if os.path.exists(pdf_file):
            with open(pdf_file, "rb") as f:
                pdf_data = f.read()
                base64_pdf = base64.b64encode(pdf_data).decode('utf-8')
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="1000" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)
            st.download_button(label="Télécharger le papier", data=pdf_data, file_name=os.path.basename(pdf_file), mime="application/pdf")
        else:
            st.write("Fichier PDF non trouvé")
def find_top_n_patches(abs_diff, n_patches, patch_size):
    top_n_patches = []
    for _ in range(n_patches):
        max_diff_index = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
        top_left_x = max_diff_index[1] - patch_size[1] // 2
        top_left_y = max_diff_index[0] - patch_size[0] // 2
        patch = abs_diff[top_left_y:top_left_y + patch_size[0], top_left_x:top_left_x + patch_size[1]]
        top_n_patches.append(((top_left_y, top_left_x), patch))
        # Set the region around the found patch to zero to avoid selecting it again
        abs_diff[top_left_y:top_left_y + patch_size[0], top_left_x:top_left_x + patch_size[1]] = 0
    return top_n_patches

st.set_page_config(
    page_title="Projet Modélisation traitement image ",
    layout="wide",
    initial_sidebar_state="expanded")
st.sidebar.title("Navigation")

page = st.sidebar.radio( "",["Introduction","Notre contribution","Principe de l'algorithme","Exemple d'exécution", "Lecture des papiers", "Démonstration","Conclusion et perspectives"])

if page == "Introduction":
    st.header("“Zero-Shot” Super-Resolution using Deep Internal Learning")
    st.subheader("Introduction")
    st.write("”Zero-Shot” Super-Resolution (ZSSR) introduces a new unsupervised CNN-based  method  for  single-image  super-resolution.    An  image-specific  CNN  istrained  on  the  low-resolution  input  image,  and  learns  to  recover  the  originalresolution from downscaled versions of the image generated by data augmen-tation,  by  taking  advantage  of  natural  regularities.   This  process  works  well in scenarios with non-ideal downscaling kernels and poor-quality low-resolutionimages (e.g. historical images, smartphone shots. . . ), and outperforms state ofthe art methods on this kind of images.")
    st.write("\n\n\n")
    left, center, right = st.columns([1,6,1])  # Adjust the ratio as needed
    with center : 
        st.image("figs/Centrale_Lille.png",width=500)
    
    st.markdown("")
    st.markdown("")
    st.markdown("")
    with st.container():
            col1,col2,col3,col4=st.columns(4)
            with col1:
                st.subheader('Benichou Yaniv')
            with col2:
                st.subheader('Bonnefoy Nicolas')
            with col3:
                st.subheader('Guckert Mathis')
            with col4:
                st.subheader('Dahy Simon')

elif page == "Notre contribution":
    st.header("Notre contribution")

    # Création des onglets
    tab1, tab2 = st.tabs(["Fix - Debug", "Features - Upgrade"])


    # Liste des bullet points pour chaque colonne
    bullet_points_col1 = [
        "Passage de python 2 à python 3",
        "Nettoyage du code, réorganisation des chemins pour chercher et ranger les images",
        "Correction du fonctionnement des configs",
        "Correction du fonctionnement avec GPU ( passage de 10 à 1 minutes en moyenne )",

    ]

    bullet_points_col2 = [
        "Ajout du traitement des JPG( originellement que PNG), y compris avec Ground Truth",
        "Ajout du fonctionnement sur 1 image au lieu d’un répertoire complet",
        "Ajout de fichiers Logs",
        "Calcul des scores et des patchs les plus différents ",
        "Streamlit pour exécuter l’algorithme  à partir d’upload, en choisissant les paramètres et un affichage dynamique."
    ]

    # Affichage des bullet points dans la colonne 1
    with tab1:
        
        for point in bullet_points_col1:
            st.markdown(f"- {point}")

    # Affichage des bullet points dans la colonne 2
    with tab2:
        
        for point in bullet_points_col2:
            st.markdown(f"- {point}")

elif page =="Principe de l'algorithme":
    st.header("Principe de l'algorithme")
    st.subheader("Entraînement d’un CNN spécifique à l’image en entrée")

    st.markdown("")
    st.image("figs/sketch.png",caption="sketch")
    st.markdown("")
    st.markdown("""
    Les données d'entraînement pour le CNN sont générées en construisant des ensembles de données spécifiques à chaque image, en redimensionnant les images basse résolution multiples fois avec des augmentations telles que des rotations et des miroirs, tandis que le CNN est entraîné à minimiser la perte L1 en ajustant le taux d'apprentissage en fonction de l'erreur de reconstruction.
    """)

elif page == "Exemple d'exécution":
    st.header("Un exemple d'exécution")
    st.subheader("Voyez-vous une différence à l'oeil nu ?")
    col1,col2 = st.columns(2)
    with col1 : 
        st.image("figs/gibon_gt.png",caption="Original Gibon")
    with col2 :
        st.image("figs/gibon_ZSSR.png",caption="ZSSR Gibon")
    st.write("\n\n\n")

    # Création des onglets
    tab1, tab2 = st.tabs(["Graphiquement", "Numériquement"])
    with tab1 : 
        st.subheader("Avec des patchs")
        col1,col2 = st.columns(2)
        with col1 : 
            st.image("figs/patch1_gt.png",caption="Original patch 1 ")
            st.write("\n")
            st.image("figs/patch2_gt.png",caption="Original patch 2 ")

        with col2 :
            st.image("figs/patch1_ZSSR.png",caption="ZSSR patch 1 ")
            st.write("\n")
            st.image("figs/patch2_ZSSR.png",caption="ZSSR patch 2 ")
    with tab2 : 
        
        col1,col2 = st.columns(2)
        with col1 : 
            st.markdown("### Structural Similarity Index (SSIM)", unsafe_allow_html=True)
            st.write("\n")
            st.markdown("""
                        Métrique classique qui permet de capturer des différences structurelles plus générale que des différences de paire de pixels. Elle se calcule sur des fenêtres de différentes tailles en utilisant les moyennes et covariances des pixels sur ces fenêtres.
                        """)
            st.write("\n")
            st.markdown("<i>Le SSIM est compris entre -1 et 1 et vaut 1 si les images sont identiques.</i>", unsafe_allow_html=True)

        with col2 :
            st.markdown("### Peak Signal-to-Noise Ratio (PSNR) ")
            st.write("\n")
            st.markdown("""
                        Où MAX est la valeur maximale pour un pixel de l’image d’origine, et MSE l’erreur quadratique moyenne entre l’image d’origine et celle reconstruite. 
                        """)
            st.write("\n")
            st.markdown("<i>Plus la valeur du PSNR est haute (en décibels) , meilleure est la reconstruction.</i>", unsafe_allow_html=True)
    
        st.markdown("""
        <style>
        .score-box {
            border: 2px solid red;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0px;  # Ajoute une marge en haut et en bas
        }
        </style>
        """, unsafe_allow_html=True)
        # Création de deux colonnes pour les scores
        col1, col2 = st.columns(2)

        # Affichage des scores dans les colonnes avec le style personnalisé
        with col1:
            # Utilisation de Markdown avec du HTML personnalisé pour le style
            st.markdown(f'<div class="score-box">PSNR = 38.6</div>', unsafe_allow_html=True)

        with col2:
            # De même pour le second score
            st.markdown(f'<div class="score-box">SSIM = 0.96</div>', unsafe_allow_html=True)
        st.markdown("\t\t\t\t ###### <i>In the examples in the paper, the authors manage to achieve PSNRs between 25 and 27, and SSIM ratios around 0.8 . </i>", unsafe_allow_html=True)

elif page == "Lecture des papiers":
    st.header("Lecture des papiers PDF")
    # Choix du mode d'affichage
    mode_affichage = st.radio("Mode d'affichage :", ["Côte à côte", "Séparé"],horizontal = True)

    # Définition des chemins des fichiers PDF et leurs noms pour la selectbox
    pdf_files = {
        "Papier original": "./Shocher et al. - 2018 - Zero-Shot Super-Resolution Using Deep Internal Lea.pdf",
        "Notre rapport": "./Report_ZSSR.pdf"  
    }

    if mode_affichage == "Côte à côte":
        col1, col2 = st.columns(2)
        with col1:
            afficher_pdf(list(pdf_files.values())[0])
        with col2:
            afficher_pdf(list(pdf_files.values())[1])
    else:  # Mode Séparé
        selected_pdf_name = st.selectbox("Rapport à afficher : ", list(pdf_files.keys()))
        afficher_pdf(pdf_files[selected_pdf_name])

elif page == "Démonstration":
    st.header("Démonstration de Zero Shot Super Resolution")
    
    # Upload an image
    uploaded_image = st.file_uploader("Upload a JPG/JPEG/PNG image", type=["jpg", "jpeg","png"])
    config_options = ['X2_ONE_JUMP_IDEAL_CONF', 'X2_IDEAL_WITH_PLOT_CONF', 'X2_GRADUAL_IDEAL_CONF', 'X2_GIVEN_KERNEL_CONF', 'X2_REAL_CONF']
    selected_config = st.selectbox("Choisissez la configuration :", config_options, index=0)
    default_image_path = None
    carousel = True
    if uploaded_image is not None:
        # Display the uploaded image
        file_size = uploaded_image.size
        max_size = 2 * 1024 * 1024  # 2 MB limit

        if file_size > max_size:
            st.error("The file you are trying to upload is too large. Please choose a file smaller than 2MB.")
            # You can also clear the uploaded file here if needed
        else:
            st.success("File uploaded successfully.")
        image = Image.open(uploaded_image)
        file_name = uploaded_image.name  # Ou spécifiez un nouveau nom de fichier
        folder_path = 'test_data'
        file_path = os.path.join(folder_path, file_name)
        # Enregistrer l'image dans le dossier spécifié
        image.save(file_path)
        st.write(f"L'image a été enregistrée dans {file_path}.")

        st.image(image, caption='Uploaded Image to be augmented.')
        uploaded_image_path = file_path
    else:
        # Display a default image
        default_image_path = os.path.join('test_data', 'gibbon.jpeg')
        if os.path.exists(default_image_path):
            image = Image.open(default_image_path)
            st.image(image, caption='Default Image to be augmented.')
        else:
            st.error("Default image not found.")

    # Button to execute the process
    if st.button('Appliquer Super Résolution'):
        log_file_name = "logs/exec_"+datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".log"
        logger.add(log_file_name, rotation="500 MB", level="TRACE")  # Rotation du fichier après 500 Mo
        if default_image_path is not None:
            logger.success(f"Image {default_image_path} correctement chargée.")
            results = run_ZSSR.main(config_options,None,default_image_path)
            logger.success(f"Fin de l'algorithme.")
        else : 
            logger.success(f"Image {uploaded_image_path} correctement chargée.")
            results = run_ZSSR.main(config_options,None,uploaded_image_path)
            logger.success(f"Fin de l'algorithme.")


        for img in results : 
            try: 
                st.image(img, caption='Processed Image')
                st.write(f"L'image augmentée a été enregistrée dans results/.")

            except:
                st.error("Processed image not found.")
        
        st.subheader("Visualization of top Patchs")
        if default_image_path is not None:
            img_original = cv2.imread(default_image_path,cv2.COLOR_BGR2RGB)
        else:
            img_original = cv2.imread(uploaded_image_path,cv2.COLOR_BGR2RGB)

        directory = "./results/result"

        # Get list of files in the directory
        files = os.listdir(directory)
        # Get modification times for each file
        modification_times = [(f, os.path.getmtime(os.path.join(directory, f))) for f in files]

        # Sort files by modification time (newest first)
        modification_times.sort(key=lambda x: x[1], reverse=True)

        # Get path of the most recently modified file
        latest_file_path = os.path.join(directory, modification_times[0][0])

        img_zssr = cv2.imread(latest_file_path,cv2.COLOR_BGR2RGB)

        img_original=cv2.resize(img_original,(img_zssr.shape[1],img_zssr.shape[0]))

        # Convert images to grayscale
        img_original_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
        img_zssr_gray = cv2.cvtColor(img_zssr, cv2.COLOR_BGR2GRAY)

        # Compute absolute difference
        abs_diff = cv2.absdiff(img_original_gray, img_zssr_gray)

        # Parameters
        n_patches = 5

        patch_size = (100, 150)  # height, width

        # Find the top n patches
        top_n_patches = find_top_n_patches(abs_diff, n_patches, patch_size)
        black_rectangle = np.zeros((patch_size[0], 10, 3), dtype=np.uint8)  # Exemple: hauteur de 100, largeur de 10 pour l'espace

        # Plot the patches
        for i, ((top_left_y, top_left_x), patch) in enumerate(top_n_patches):
            #dispaly the pairs of patches side by side with a legend per image
            st.write(f"Top patch {i + 1}")
            st.image([cv2.cvtColor(img_original[top_left_y:top_left_y + patch_size[0], top_left_x:top_left_x + patch_size[1]],
                                    cv2.COLOR_BGR2RGB),cv2.cvtColor(img_zssr[top_left_y:top_left_y + patch_size[0], top_left_x:top_left_x + patch_size[1]],cv2.COLOR_BGR2RGB)], caption=["Original", "ZSSR"])

elif page == "Conclusion et perspectives":
    st.header("Conclusion et perspectives d'améliorations")
    st.balloons()
    st.markdown(f"- Approche simple, mais avec quelques subtilités ")
    st.markdown(f"- Simulations limitées, mais beaucoup d'hyperparamètres et de configurations à tester, certaines non implémentées (estimation du noyau de réduction d'échelle) ")
    st.markdown(f"- Cas d'utilisations assez spécifique car temps d'exécution long par image ")
    #st.write(f"\n\n\n\n\n\n")
    
    st.subheader(" \t\t\t\tAvez-vous des questions ? ")

