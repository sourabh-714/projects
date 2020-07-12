from tensorflow import keras
model = keras.models.load_model('/content/drive/My Drive/my_model.h5')

predict_dir_path='/content/drive/My Drive/class/mix/'
onlyfiles = [f for f in listdir(predict_dir_path) if isfile(join(predict_dir_path, f))]
print(onlyfiles)

# predicting images
from keras.preprocessing import image
i_counter = 0 
r_counter  = 0
for file in onlyfiles:
    img = image.load_img(predict_dir_path+file, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    
    images = np.vstack([x])
    classes = model.predict_classes(images, batch_size=10)
    classes = classes[0][0]
    
    if classes == 0:
        print(file + ": " + 'invoices')
        i_counter += 1
    else:
        print(file + ": " + 'receipts')
        r_counter += 1
print("invoices :",i_counter)
print("Total receipts :",r_counter)