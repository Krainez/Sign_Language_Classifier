import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)
time.sleep(2)

def process_image(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image1 = cv2.resize(image1, (28,28), interpolation = cv2.INTER_CUBIC)
    image1=np.expand_dims(image1,axis=0)
    image1=np.expand_dims(image1,axis=3)
    return image1
    
def predict_gesture(model):
    while(True):
    
    # Capture frame-by-frame
        ret, frame_im = cap.read()
        roi=frame_im[100:350,100:350]
        cv2.rectangle(frame_im,(100,100),(350,350),(0,255,0),0)
    # Our operations on the frame come here
        p_image=process_image(roi)
        result=model.predict(p_image)
        
        cv2.putText(frame_im,result,(40,40),cv2.FONT_HERSHEY_TRIPLEX,2,255)
    # Display the resulting frame
        cv2.imshow('frame',frame_im)
        #cv2.imshow('mask',roi)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
