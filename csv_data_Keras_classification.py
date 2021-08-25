import os
os.environ['TF_cpp_MIN_LEVEL'] =  '2'

import pandas as pd
import numpy as np
# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import IntegerLookup
from tensorflow.keras.layers import Normalization
from tensorflow.keras.layers import StringLookup

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def check_csv_contents(file):
    dataframe = pd.read_csv(file)
    # print(f'Top5 datas: \n{df.head()}')
    # print(f'Last5 datas: \n{df.tail()}')
    # print(f'shape: {df.shape}')
    return dataframe


def df_to_dataset(dataframe, target):
    dataframe = dataframe.copy()
    labels = dataframe.pop(target)

    # 使用tf.data.Dataset.from_tensor_slices()方法，我們可以獲取列表或數組的切片。
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    # shuffle(): 用來打亂數據集中數據順序.
    # buffer_size: https://codertw.com/%E7%A8%8B%E5%BC%8F%E8%AA%9E%E8%A8%80/661458/
    ds = ds.shuffle(buffer_size=len(dataframe))

    return ds


def encode_numerical_feature(feature, name, dataset):
    # Create a Normalization layer for our feature
    normalizer = Normalization()

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the statistics of the data
    normalizer.adapt(feature_ds)

    # Normalize the input feature
    encoded_feature = normalizer(feature)
    return encoded_feature


def encode_categorical_feature(feature, name, dataset, is_string):
    lookup_class = StringLookup if is_string else IntegerLookup
    # Create a lookup layer which will turn strings into integer indices
    lookup = lookup_class(output_mode="binary")

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the set of possible string values and assign them a fixed integer index
    lookup.adapt(feature_ds)

    # Turn the string input into integer indices
    encoded_feature = lookup(feature)
    return encoded_feature


if __name__ == '__main__':

    # Structured data classification from scratch.
    # From: https://keras.io/examples/structured_data/structured_data_classification_from_scratch/
    #*************************************************#
    #               Prepare Dataset.                  #                           
    #*************************************************#
    dataset_csv_file = './datasets/heart.csv'
    df = check_csv_contents(file=dataset_csv_file)
    target_value = "target"

    # frac(float): 要抽出的比例, random_state：隨機的狀態.
    val_df = df.sample(frac=0.2, random_state=1337)
    # drop the colum 1 of 'class'.
    train_df = df.drop(val_df.index)
    # print(f'\nlen of: \ndf: {len(df)}, train_df:{len(train_df)}, val_df: {len(val_df)}')

    train_ds = df_to_dataset(dataframe=train_df, target=target_value)
    val_ds = df_to_dataset(dataframe=val_df, target=target_value)

    # # .take(n): get n datas.
    # for x, y in train_ds.take(1):
    #     # tf.print("Input(Features):", x)
    #     tf.print("Target:", y)

    train_ds = train_ds.batch(32)
    val_ds = val_ds.batch(32)

    #*************************************************#
    #               Build a model.                    #                           
    #*************************************************#
    # Categorical features encoded as integers
    sex = keras.Input(shape=(1,), name="sex", dtype="int64")
    cp = keras.Input(shape=(1,), name="cp", dtype="int64")
    fbs = keras.Input(shape=(1,), name="fbs", dtype="int64")
    restecg = keras.Input(shape=(1,), name="restecg", dtype="int64")
    exang = keras.Input(shape=(1,), name="exang", dtype="int64")
    ca = keras.Input(shape=(1,), name="ca", dtype="int64")

    # Categorical feature encoded as string
    thal = keras.Input(shape=(1,), name="thal", dtype="string")

    # Numerical features
    age = keras.Input(shape=(1,), name="age")
    trestbps = keras.Input(shape=(1,), name="trestbps")
    chol = keras.Input(shape=(1,), name="chol")
    thalach = keras.Input(shape=(1,), name="thalach")
    oldpeak = keras.Input(shape=(1,), name="oldpeak")
    slope = keras.Input(shape=(1,), name="slope")
    
    all_inputs = [
        sex,
        cp,
        fbs,
        restecg,
        exang,
        ca,
        thal,
        age,
        trestbps,
        chol,
        thalach,
        oldpeak,
        slope,
    ]

    # Integer categorical features
    sex_encoded = encode_categorical_feature(sex, "sex", train_ds, False)
    cp_encoded = encode_categorical_feature(cp, "cp", train_ds, False)
    fbs_encoded = encode_categorical_feature(fbs, "fbs", train_ds, False)
    restecg_encoded = encode_categorical_feature(restecg, "restecg", train_ds, False)
    exang_encoded = encode_categorical_feature(exang, "exang", train_ds, False)
    ca_encoded = encode_categorical_feature(ca, "ca", train_ds, False)

    # String categorical features
    thal_encoded = encode_categorical_feature(thal, "thal", train_ds, True)

    # Numerical features
    age_encoded = encode_numerical_feature(age, "age", train_ds)
    trestbps_encoded = encode_numerical_feature(trestbps, "trestbps", train_ds)
    chol_encoded = encode_numerical_feature(chol, "chol", train_ds)
    thalach_encoded = encode_numerical_feature(thalach, "thalach", train_ds)
    oldpeak_encoded = encode_numerical_feature(oldpeak, "oldpeak", train_ds)
    slope_encoded = encode_numerical_feature(slope, "slope", train_ds)

    all_features = layers.concatenate(
        [
            sex_encoded,
            cp_encoded,
            fbs_encoded,
            restecg_encoded,
            exang_encoded,
            slope_encoded,
            ca_encoded,
            thal_encoded,
            age_encoded,
            trestbps_encoded,
            chol_encoded,
            thalach_encoded,
            oldpeak_encoded,
        ]
    )
    
    x = layers.Dense(32, activation="relu")(all_features)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(all_inputs, output)
    print(f'model: {model}')
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    print(f'compile: {model}')

    #*************************************************#
    #               Train the model.                  #                           
    #*************************************************#
    model.fit(x=train_ds, epochs=50, verbose=1, validation_data=val_ds)

    # Save modle
    all_model = './model_weights/all_model/08.25/heart_disease_origin'
    model.save(all_model)
    print("All model: save done! \n")

    # checkpoint_path = 'model_weights/ckpt/08.20/heart_disease'
    # model.save_weights(checkpoint_path)
    # print("ckpt:  save done!")

    # print(model.summary())

    import sys
    sys.exit()
    #*************************************************#
    #               Inference on new data.            #                           
    #*************************************************#
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

    input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}
    predictions = model.predict(input_dict)


    print("This particular patient had a %.1f percent probability "
        "of having a heart disease, as evaluated by our model." % (100 * predictions[0][0],)
    )

    print(f'heart disease prob: {round(predictions[0][0]*100, 2)} %')