import cv2
from ultralytics import YOLO

model = YOLO("yolov8m-seg.pt")
kaynak = cv2.VideoCapture(0)      # 0 = varsayılan kamera. Video için: "video.avi"

while True:
    ok, kare = kaynak.read()
    if not ok:
        break
    sonuc = model(kare, conf=0.4, verbose=False)[0]
    cizim = sonuc.plot()          # kutu + maske + etiket çizilmiş kare
    cv2.imshow("Tespit (q ile cik)", cizim)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

kaynak.release()
cv2.destroyAllWindows()