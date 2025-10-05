# --- LIBRERIAS---
import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Detector Facial con Lentes y Bigote - OpenCV + Streamlit")

st.write("""
Sube una imagen y el sistema:
1. Detectará el **rostro**  
2. Luego los **ojos** (para los lentes)  
3. Detectará la **boca** (para el bigote)
""")

# --- Clasificadores Haar ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# --- aqui se cargan los lentes y bigote ---
glasses_img = cv2.imread('overlays/glasses.png', cv2.IMREAD_UNCHANGED)
moustache_img = cv2.imread('overlays/moustache.png', cv2.IMREAD_UNCHANGED)

# --- subir una imagen de rostro
uploaded_file = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns(2)
add_glasses = col1.button("Agregar Lentes")
add_moustache = col2.button("Agregar Bigote")

# --- funcion para superponer imagenes PNG ---
def overlay_image(background, overlay, x, y, w, h):
    """Superpone un PNG transparente sobre la imagen base."""
    overlay_resized = cv2.resize(overlay, (w, h))
    if overlay_resized.shape[2] == 4:
        alpha_overlay = overlay_resized[:, :, 3] / 255.0
        alpha_background = 1.0 - alpha_overlay

        for c in range(3):
            y1, y2 = max(y, 0), min(y + h, background.shape[0])
            x1, x2 = max(x, 0), min(x + w, background.shape[1])

            overlay_crop = overlay_resized[:y2 - y1, :x2 - x1, c]
            alpha_o = alpha_overlay[:y2 - y1, :x2 - x1]
            alpha_b = alpha_background[:y2 - y1, :x2 - x1]

            background[y1:y2, x1:x2, c] = (
                alpha_o * overlay_crop + alpha_b * background[y1:y2, x1:x2, c]
            )
    return background

# --- proceso principal ---
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    st.write(f"Se detectaron {len(faces)} rostro(s).")

    result_img = img_array.copy() 

    # si no se elige ningun accesorio, solo se muestra la imagen original
    if not add_glasses and not add_moustache:
        st.image(result_img, caption="Imagen original", use_container_width=True)
    else:
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = result_img[y:y + h, x:x + w]

            # --- AQUI SE AGREGAN LOS LENTES ---
            if add_glasses:
                eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 10)
                if len(eyes) >= 2:
                    eyes = sorted(eyes, key=lambda ex: ex[0])
                    x1, y1, w1, h1 = eyes[0]
                    x2, y2, w2, h2 = eyes[1]

                    ex = min(x1, x2)
                    ey = min(y1, y2) - int(h1 * 0.3)
                    ew = max(x1 + w1, x2 + w2) - ex
                    ew = int(ew * 1.6)
                    eh = int(max(h1, h2) * 1.7)
                    ex = max(ex - int(ew * 0.15), 0)
                    ey = max(ey, 0)

                    roi_color = overlay_image(roi_color, glasses_img, ex, ey, ew, eh)

            # --- AQUI SE AGREGA EL BIGOTE ---
            if add_moustache:
                mouths = mouth_cascade.detectMultiScale(roi_gray, 1.5, 15)
                for (mx, my, mw, mh) in mouths:
                    if my > h / 2:  
                        mw_bigote = int(mw * 1.3)
                        mh_bigote = int(mh * 0.8)
                        mx_bigote = mx - int((mw_bigote - mw) / 2)
                        my_bigote = my - int(mh * 0.25)  
                        roi_color = overlay_image(
                            roi_color, moustache_img, mx_bigote, my_bigote, mw_bigote, mh_bigote
                        )
                        break

            result_img[y:y + h, x:x + w] = roi_color

        st.image(result_img, caption="Imagen procesada", use_container_width=True)
