import cv2
import numpy as np

# Buka video
cap = cv2.VideoCapture(r"C:\project\uro\code\object_video.mp4")

#      Definisikan batas bawah dan atas warna dalam HSV 
lower_bound = np.array([155, 25, 0])  # HSV lower bound
upper_bound = np.array([179, 255, 255])  # HSV upper bound
min_area = 500  # Area minimum untuk objek yang ingin di-track

# Membaca setiap frame dari video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # Konversi frame dari RGB ke HSV
    mask = cv2.inRange(hsv, lower_bound, upper_bound) # Membuat mask
    
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # Temukan kontur dari mask

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area: # Filter berdasarkan area
            x, y, w, h = cv2.boundingRect(contour) # Dapatkan bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2) # Gambarkan rectangle di sekitar objek
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2) # Opsional: gambarkan kontur

    cv2.imshow('Detected Objects', frame) # Menampilkan hasil
    cv2.imshow('Mask', mask)

    if cv2.waitKey(10) & 0xFF == ord('q'): # Tekan 'q' untuk keluar dari tampilan
        break

# Bersihkan dan tutup semua jendela
cap.release()
cv2.destroyAllWindows()