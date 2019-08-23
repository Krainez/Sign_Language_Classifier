from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from data_read import TRAIN,TEST,labels_train,labels_test,input_shape
import numpy as np


class ai_model:
    def __init__(self,num_classes,batch_size,epoch):
        self.num_class=num_classes
        self.batch_size=batch_size
        self.epoch=epoch
        self.model=Sequential()
    
    def train_model(self):
        self.model.add(Conv2D(64, kernel_size=(3,3), activation='relu', input_shape=input_shape))
        self.model.add(MaxPooling2D(pool_size = (2, 2)))
        self.model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size = (2, 2)))
        self.model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size = (2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(128, activation = 'relu'))
        self.model.add(Dropout(0.20))
        self.model.add(Dense(self.num_class, activation = 'softmax'))

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model_data=self.model.fit(TRAIN, labels_train, validation_data=(TEST, labels_test), epochs=self.epoch, batch_size=self.batch_size)
    
        return model_data
    def predict(self,img):
        alphabet={0:"A",1:"B",2:"C",3:"D",4:"E",5:"F",6:"G",7:"H",8:"I",9:"K",10:"L",11:"M",12:"N",13:"O",14:"P",15:"Q",16:"R",17:"S",18:"T",19:"U",20:"V",21:"W",22:"X",23:"Y"}
        ans=self.model.predict(img)
        maximum=np.argmax(ans)
        return alphabet.get(maximum,"Invalid")
