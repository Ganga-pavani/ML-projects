import numpy as np 
from keras.preprocessing import image 
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions 
model = VGG16(weights='imagenet') 
img_path = 'puppy.jpg' 
img = image.load_img(img_path, target_size=(224, 224)) 
img_array = image.img_to_array(img) 
img_array = np.expand_dims(img_array, axis=0) 
img_array = preprocess_input(img_array) 
predictions = model.predict(img_array) 
decoded_predictions = decode_predictions(predictions, top=3)[0]  # Get top 3 predicted labels 
print("Top predictions:") 
for i, (imagenet_id, label, score) in enumerate(decoded_predictions): 
    print(f"{i + 1}: {label} ({score:.2f})")
