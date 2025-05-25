import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from fastapi import FastAPI
from tensorflow.keras import layers
from tensorflow.keras import regularizers

#Hauptquelle: https://www.tensorflow.org/tutorials/keras/regression

#https://www.tensorflow.org/tutorials/keras/regression#regression_with_a_deep_neural_network_dnn
#https://www.tensorflow.org/api_docs/python/tf/keras/layers/Rescaling
#https://datascience.stackexchange.com/questions/65995/tensorflow-sigmoid-activation-function-as-output-layer-value-interpretation
def build_and_compile_model(norm):
    model = tf.keras.Sequential([
        norm,
        layers.Dense(32, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid'),
        tf.keras.layers.Rescaling(scale=9, offset=1)
    ])
    
    model.compile(
        loss='mean_absolute_error',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
    )
    return model

#https://www.tensorflow.org/tutorials/keras/regression#linear_regression
#https://hellocoding.de/blog/coding-language/python/csv-lesen-schreiben
#https://www.tensorflow.org/tutorials/keras/regression#conclusion
def plot_print_and_save_history(history):
    plt.clf()
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 15])
    plt.xlabel('Epoche')
    plt.ylabel('Fehler')
    plt.legend()
    plt.grid(True)
    plt.savefig('../Results/loss.png')
    plt.close()
    print(f"Loss: {history.history['loss'][-1]}")
    print(f"Validation Loss: {history.history['val_loss'][-1]}")
    
    results = [
        {"name": "Epochen", "values": len(history.history['loss'])},
        {"name": "Loss", "values": history.history['loss'][-1]},
        {"name": "Validation Loss", "values": history.history['val_loss'][-1]}
    ]
    
    results_dataset = pd.DataFrame(results, columns=["name", "values"])
    results_dataset.to_csv("../Results/historyValues.csv", index=False, encoding="utf-8")

def predict_and_filter_top3(model, dataset_unrated_encoded, dataset_unrated, column_names):
    #https://www.tensorflow.org/tutorials/keras/regression#make_predictions
    dataset_unrated_features = dataset_unrated_encoded[column_names].astype(np.float32).values
    predicted_ratings = model.predict(dataset_unrated_features).flatten()
    dataset_unrated['predicted_rating'] = predicted_ratings

    #https://www.usepandas.com/csv/sort-csv-data-by-column
    top_recommendations = dataset_unrated.sort_values(by='predicted_rating', ascending=False).head(3)
    top_recommendations.to_csv("../Results/top_recommendations.csv", index=False)

    for i, row in top_recommendations.iterrows():
        print(f"Empfehlung: {row['Title']}")
        print(f"Vorhersage-Bewertung: {row['predicted_rating']}")

    dataset_unrated_encoded.to_csv("../Data/dataset_unrated_encoded_with_predictions.csv", index=False)

def do_ml_process():
    #https://www.tensorflow.org/tutorials/load_data/csv
    dataset_rated = pd.read_csv("../Data/UserDataRating.csv")
    dataset_unrated = pd.read_csv("../Data/MovieData.csv")

    #https://tech-champion.com/programming/python-programming/pandas-one-hot-encoding-combining-datasets-for-comprehensive-analysis/?utm_source=chatgpt.com#google_vignette
    dataset_combined = pd.concat([dataset_rated, dataset_unrated], ignore_index=False)

    #https://www.tensorflow.org/tutorials/keras/regression#get_the_data
    dataset_encoded = pd.get_dummies(dataset_combined, columns=['Typ', 'Genre', 'Topic'])

    dataset_rated_encoded = dataset_encoded.iloc[:len(dataset_rated)].copy()
    dataset_unrated_encoded = dataset_encoded.iloc[len(dataset_rated):].copy()

    column_names = [col for col in dataset_rated_encoded.columns if col not in ['Title', 'Rating']]

    #https://www.tensorflow.org/tutorials/keras/regression#split_the_data_into_training_and_test_sets
    train_dataset = dataset_rated_encoded.sample(frac=0.8, random_state=0)
    test_dataset = dataset_rated_encoded.drop(train_dataset.index)

    train_features = train_dataset.copy()[column_names].astype(np.float32)
    test_features = test_dataset.copy()[column_names].astype(np.float32)

    train_labels = train_dataset['Rating'].astype(np.float32)
    test_labels = test_dataset['Rating'].astype(np.float32)


    #https://www.tensorflow.org/tutorials/keras/regression#the_normalization_layer
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.array(train_features))
    
    
    model = build_and_compile_model(normalizer)
    
    #https://medium.com/@piyushkashyap045/early-stopping-in-deep-learning-a-simple-guide-to-prevent-overfitting-1073f56b493e
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    numberOfmaximumEpochs = 50
    history = model.fit(
        train_features, train_labels,
        validation_split=0.2,
        epochs=numberOfmaximumEpochs,
        callbacks=[callback],
        verbose=0
    )
    
    plot_print_and_save_history(history)
    predict_and_filter_top3(model, dataset_unrated_encoded, dataset_unrated, column_names)
    
    #https://www.tensorflow.org/tutorials/keras/save_and_load
    model.save("../Data/modell.keras")
    
def main():
  do_ml_process()

if __name__ == "__main__":
    main()

