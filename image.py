import cv2 as cv
image = cv.imread('who.jpg')
cv.imshow("Original Image", image 

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow("Grayscale Image", gray)

haar_face = cv.CascadeClassifier('haar_face.xml')

scaleFactor = 1.1
minNeighbors = 3 

face_rect = haar_face.detectMultiScale(gray, scaleFactor, minNeighbors)
print(f'The number of faces detected is... {len(face_rect)}')

for (x, y, w, h) in face_rect:
    cv.rectangle(image, (x, y), (x+w, y+h), (37, 66, 0), thickness=2)


cv.imshow("Detected Faces", image)

cv.waitKey(0)
cv.destroyAllWindows()
