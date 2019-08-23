from ai_model import ai_model
import matplotlib.pyplot as plt
from capture_image import predict_gesture
from keras.models import load_model
import os.path


if os.path.exists('./Sign_language_predicter.h5'):
    predicter=ai_model(24,128,30)
    predicter.model=load_model('Sign_language_predicter.h5')
    print("Loaded model from disk")
else:
    predicter=ai_model(24,128,30)
    model_data=predicter.train_model()
    plt.plot(model_data.history['acc'])
    plt.plot(model_data.history['val_acc'])
    plt.title("Accuracy")
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['train','test'])
    plt.show()
    predicter.model.save('Sign_language_predicter.h5')
    print("Saved model to disk")
    
predict_gesture(predicter)