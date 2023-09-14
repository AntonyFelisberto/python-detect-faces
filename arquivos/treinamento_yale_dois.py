import cv2
import os
import numpy as np
from PIL import Image

detector_face = cv2.CascadeClassifier("arquivos\\arquivo_faces\\haarcascade_frontalface_default.xml")
reconhecedor = cv2.face.EigenFaceRecognizer_create()
reconhecedor.read("classificadorEigenYale.yml")

total_acertos = 0
percentual_acertos = 0
total_confianca = 0.0

caminhos = [os.path.join('arquivos\\yalefaces\\teste', f) for f in os.listdir('arquivos\\yalefaces\\teste')]
for caminhos_imagem in caminhos:
    imagem_face = Image.open(caminhos_imagem).convert("L")
    imagem_face_NP = np.array(imagem_face,"uint8")
    faces_detectadas = detector_face.detectMultiScale(imagem_face_NP)
    for (x,y,l,a) in faces_detectadas:
        id_previsto, confianca = reconhecedor.predict(imagem_face_NP)
        id_atual = int(os.path.split(caminhos_imagem)[1].split(".")[0].replace("subject",""))
        if id_previsto == id_atual:
            total_acertos += 1
            total_confianca += confianca
        cv2.rectangle(imagem_face_NP,(x,y),(x + l,y+a),(0,0,255),2)
        cv2.waitKey(1000)

pontual_acertos = (total_acertos/30) * 100
total_confianca = total_confianca / total_acertos
