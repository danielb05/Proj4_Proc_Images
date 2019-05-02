import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob

# função que utiliza o simple Blob detector para identificar o valor do dado
def valorDado(img):
    detector = cv2.SimpleBlobDetector_create()
    keypoints = detector.detect(img)
    return len(keypoints)

# importando fonte utilizada
font = cv2.FONT_HERSHEY_SIMPLEX

cap = cv2.VideoCapture('dados2.mp4')

while cap.isOpened() :
    ret, frame = cap.read()

    # lendo a imagem em escala de cinza
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # fazendo o threshold (binário preto e branco)
    ret, thresh = cv2.threshold(img, 245, 255, cv2.THRESH_BINARY_INV)

    # reduzindo ruídos na imagem
    medianFiltered = cv2.medianBlur(thresh, 5)

    # encontra os contornos de figuras detectadas na imagem
    contours, hierarchy = cv2.findContours(medianFiltered, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # lista de contornos válidos
    contour_list = []

    # verifica nos contornos encontrados se o mesmo possui area maior q 100px (contorno válido)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:
            contour_list.append(contour)

    # converte a imagem de escala de cinza para RGB
    img_cor = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # itera pela lista de contornos válidos
    for contour in contour_list:

        # pega as coordenadas do menor retângulo ao redor contorno sendo analisado
        x, y, w, h = cv2.boundingRect(contour)

        # faz uma cópia da imagem detectada dentro do retângulo
        roi = img_cor[y:y+h, x:x+w]

        # escreve o valor retornados da função valorDado na imagem
        cv2.putText(img_cor, str(valorDado(roi)), (x+w, y+h), font, 1, (0,0,255), thickness=3)

    # exibe a imagem final já com os valores dos dados detectados
    cv2.imshow('Imagem Final', img_cor)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()