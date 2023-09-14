import cv2

classificador = cv2.CascadeClassifier("arquivos\\arquivo_faces\\haarcascade_frontalface_default.xml")
classificador_olhos = cv2.CascadeClassifier("arquivos\\arquivo_faces\\haarcascade_eye.xml")
reconhecedor = cv2.face.EigenFaceRecognizer_create()
reconhecedor.read("classificadorEigen.yml")
largura, altura = 220,220
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
camera = cv2.VideoCapture(0)

while True:
    conectado, imagem = camera.read()
    imagem_cinza = cv2.cvtColor(imagem,cv2.COLOR_BGR2GRAY)
    faces_detectadas = classificador.detectMultiScale(imagem_cinza,
                                                    scaleFactor=1.5,
                                                    minSize=(30,30))
    for (x,y,l,a) in faces_detectadas:
        imagem_face = cv2.resize(imagem_cinza[y:y + a,x:x + l],(0,0,255),2)
        id, confianca = reconhecedor.predict(imagem_face)
        nome = ""
        if id == 1:
            nome = "an"
        elif id == 2:
            nome = "or"

        cv2.putText(imagem,f"{id}_{nome}",(x,y + (a+30)), font, 2 (0,0,255))
        cv2.putText(imagem,f"{confianca}",(x,y + (a+50)), font, 1 (0,0,255))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.waitKey(1)

camera.release()
cv2.destroyAllWindows()