from flask import Flask, request, jsonify
from keras.layers import Input, Embedding, Flatten, Dot, Dense, Dropout
from keras.models import Model,model_from_json,load_model
from keras.callbacks import EarlyStopping
from sklearn.externals import joblib
import requests
import json
import numpy as np
import tensorflow as tf
import pandas as pd

from keras import backend as K
app = Flask(__name__)


@app.route("/")
def hello():
    return "Hello Wrld!"


@app.route('/train', methods=['POST'])
def train():
    # get parameters from request
    parameters = request.json
    
    # For each sample we input the integer identifiers
    # of a single user and a single item
    user_id_input = Input(shape=[1],name='user')
    item_id_input = Input(shape=[1], name='item')

    embedding_size = 30
    user_embedding = Embedding(output_dim=embedding_size, input_dim=parameters['nb_users']+1 ,
                               input_length=1, name='user_embedding')(user_id_input)

    item_embedding = Embedding(output_dim=embedding_size, input_dim=parameters['nb_items']+1,
                               input_length=1, name='item_embedding')(item_id_input)

    # reshape from shape: (batch_size, input_length, embedding_size)
    # to shape: (batch_size, input_length * embedding_size) which is
    # equal to shape: (batch_size, embedding_size)
    user_vecs = Flatten()(user_embedding)
    item_vecs = Flatten()(item_embedding)

    y = Dot(axes=1)([user_vecs, item_vecs])
    x = Dense(64, activation='relu')(y)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='relu')(x)
    y = Dense(1)(x)
    model = Model(inputs=[user_id_input, item_id_input], outputs=y)
    model.compile(optimizer="adam", loss="MAE")

    # fit model
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    history = model.fit([parameters["user_history"], parameters["item_history"]], parameters["rating_history"],
                    batch_size=64, epochs=20, validation_split=0.1,
                    shuffle=True, callbacks=[early_stopping])
    # persist model
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    model.save("games.h5")
    return "Saved model to disk"
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    parameters = request.json
    # load the model, and pass in the custom metric function
    #global graph
    #graph = tf.get_default_graph()
    K.clear_session()
    model = load_model('games.h5')
    # load json and create model
    #json_file = open('model.json', 'r')
    #loaded_model_json = json_file.read()
    #json_file.close()
    #loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    #loaded_model.load_weights("model.h5")
    #print("Loaded model from disk")
    #loaded_model.compile(loss='MAE', optimizer='adam')
    user=parameters['user']
    item=parameters['item']
    rating=model.predict([np.array([user]), np.array([item])] )
    data=dict()
    data["rating"] = str(rating[0][0])
    return jsonify(data)
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000)