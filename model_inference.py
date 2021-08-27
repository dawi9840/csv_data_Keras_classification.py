import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers

if __name__ == '__main__':

    all_model = './model_weights/all_model/08.25/heart_disease_modify'
    # all_model = './model_weights/all_model/08.25/heart_disease_origin'
 
    # Loads the model and training weights.
    model = keras.models.load_model(all_model)

    # print(model.summary())

    sample = {
            "age": 60,
            "sex": 1,
            "cp": 1,
            "trestbps": 145,
            "chol": 233,
            "fbs": 1,
            "restecg": 2,
            "thalach": 150,
            "exang": 0,
            "oldpeak": 2.3,
            "slope": 3,
            "ca": 0,
            "thal": "fixed",
    }
    
    sample2 = {
        "age": 67,
        "sex": 1,
        "cp": 4,
        "trestbps": 160,
        "chol": 286,
        "fbs": 0,
        "restecg": 2,
        "thalach": 108,
        "exang": 1,
        "oldpeak": 1.5,
        "slope": 2,
        "ca": 3,
        "thal": "normal",
    }

    input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample2.items()}

    outputs = model.predict(input_dict)

    heart_disease = 'heart_disease'

    print('*'*30)
    print(f'class: {heart_disease}, prob: {round(outputs[0][0]*100, 2)} %')
