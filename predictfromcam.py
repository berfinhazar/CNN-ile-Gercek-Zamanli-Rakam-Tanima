import cv2
import numpy as np
from PIL.ImageOps import grayscale
from tensorflow.keras.models import load_model


model=load_model("mnist_cnn_v1.h5")

#kamerayı başlatmak için
cap=cv2.VideoCapture(0)

print("bir kağıda kalem ile rakam yaz ve kameraya göster, çıkmak için q tuşuna bas ")

#kameradan gelen görüntüleri ccv ile tahmin et
while True:
    success,frame=cap.read() #frame kamera görüntüleri
    if not success:
        break

    # görüntüyü gray scale hale getir
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ROI alanı çiz(kutu)
    h,w=gray.shape
    box_size=200
    top_left=(w//2 - box_size//2,h//2-box_size//2)
    bottom_right=(w//2 + box_size//2,h//2+box_size//2)
    cv2.rectangle(frame,top_left,bottom_right,(0,255,0),2)

    #roiden sayı tahmini
    roi=gray[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0]]
    roi=cv2.resize(roi,(28,28))
    roi=roi.astype("float32")/255.0
    roi=roi.reshape(1,28,28,1)

    #tahmin etme
    pred=model.predict(roi,verbose=0)
    digit=np.argmax(pred)

    #tahmini ekrana yazdırma
    cv2.putText(frame, f"Tahmin: {digit}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5 , (0, 0, 255), 2)

    cv2.imshow("Tahmin Ekranı: ",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break





