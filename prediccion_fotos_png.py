import numpy as np
from keras.models import load_model
from PIL import Image                                                # libreria Pillow
from tensorflow.keras.preprocessing.image import img_to_array


path_img = "B/093441-307-B.png"                             
img = Image.open(path_img,mode='r')                                 # Leo la imagen en el path indicado

def get_model():
    global model                                                     # guardo el modelo en varible global
    model = load_model('camiones_model_full.h5')
    print(" * Model loaded!")

print(" * Loading Keras model...")
get_model()                                                         # cargo el modelo una unica vez 


def preprocess_image(image):                                         # realiza el preprocesamiento ()
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((240, 352))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image.astype('float32')
    image /= 255
    print("* Preprocess_done")
    return image




processed_image = preprocess_image(img)                             # Aplico preprocesamiento


prediction = model.predict(processed_image).tolist()                # Etapa de prediccion

# Imprimo resultados
print("Porcentaje de cada categoria: ")
print("Camion con material","Nada","Otro","Tolva vacia")
print(np.round(np.array(prediction[0])*100.0/sum(np.array(prediction[0])),2))
#img.show()
