import cv2
import numpy as np

classificador = cv2.CascadeClassifier("arquivos\\arquivo_faces\\haarcascade_frontalface_default.xml")
classificador_olhos = cv2.CascadeClassifier("arquivos\\arquivo_faces\\haarcascade_eye.xml")
camera = cv2.VideoCapture(0)
amostra = 1
numero_amostra = 25
id = input("digite seu id: ")
largura,altura = 220,220
print("capturanndo faces")

while True:
    conectado,imagem = camera.read()
    imagem_cinza = cv2.cvtColor(imagem,cv2.COLOR_BGR2GRAY)
    faces_detectadas = classificador.detectMultiScale(imagem_cinza,
                                                      scaleFactor=1.5,
                                                      minSize=(150,150))
    for (x,y,l,a) in faces_detectadas:
        cv2.rectangle(imagem, (x,y), (x+l,y+a),(0,0,255),2)
        regiao = imagem[y:y + a,x:x + l]
        regiao_olho_cinza = cv2.cvtColor(regiao,cv2.COLOR_BGR2GRAY)
        olhos_detectados = classificador_olhos.detectMultiScale(regiao_olho_cinza)
        for (ox,oy,ol,oa) in olhos_detectados:
            cv2.rectangle(regiao,(ox,oy),(ox + ol, oy + oa), (0,255,0),2)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                if np.average(imagem_cinza) > 110:
                    imagem_face = cv2.resize(imagem_cinza[y:y + a, x:x +l],(largura,altura))
                    cv2.imwrite("arquivos\\fotos\\pessoas"+str(id)+"."+str(amostra)+"jpg",imagem_face)
                    print("foto "+str(amostra)+" efetuada com sucesso")
                    amostra +=1
    #cv2.waitKey(1)
    if amostra >= numero_amostra + 1:
        break

camera.release()
cv2.destroyAllWindows()