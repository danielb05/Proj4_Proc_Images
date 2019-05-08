import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob

# função que utiliza o simple Blob detector para identificar o valor do dado
def valorDado(img):
    
    height, width = img.shape
    img = cv2.resize(img,(height*3,width*3))
    detector = cv2.SimpleBlobDetector_create()
    keypoints = detector.detect(img)
    #cv2.imshow('Img', img)
    return len(keypoints)

# importando fonte utilizada
font = cv2.FONT_HERSHEY_SIMPLEX

#cap = cv2.VideoCapture('dados2.mp4') # vídeo em baixa qualidade
cap = cv2.VideoCapture('dados.avi') # vídeo em alta qualidade


while cap.isOpened() :
    ret, frame = cap.read()

    # reduzindo ruídos na imagem
    blurred = cv2.GaussianBlur(frame, (5,5), 0)

    
    #medianFiltered = cv2.medianBlur(blurred, 5)

    # lendo a imagem em escala de cinza
    img = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)   

    # fazendo o threshold (binário preto e branco)
    ret, thresh = cv2.threshold(img, 210, 255, cv2.THRESH_BINARY)  

    # encontra os contornos de figuras detectadas na imagem
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # lista de contornos válidos
    contour_list = []

    # verifica nos contornos encontrados se o mesmo possui area maior q 100px (contorno válido)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:
            contour_list.append(contour)

    # itera pela lista de contornos válidos
    for contour in contour_list:

        # pega as coordenadas do menor retângulo ao redor contorno sendo analisado
        x, y, w, h = cv2.boundingRect(contour)
        
        # faz uma cópia da imagem detectada dentro do retângulo
        roi = thresh[y:y+h, x:x+w]

        #cv2.imshow('Img2', thresh)

        # escreve o valor retornados da função valorDado na imagem
        val = valorDado(roi)
        if(val > 0):
            cv2.rectangle(frame,(x,y),(x+w,y+h),(200,0,0),2)
            cv2.putText(frame, str(val), (x+w, y+h), font, 1, (0,0,255), thickness=3)

    # exibe a imagem final já com os valores dos dados detectados
    cv2.imshow('Imagem Final', frame)
    
    if cv2.waitKey(15) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()