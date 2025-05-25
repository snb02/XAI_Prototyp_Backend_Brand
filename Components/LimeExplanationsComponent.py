import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
from lime.lime_tabular import LimeTabularExplainer
import tensorflow as tf

#https://www.tensorflow.org/tutorials/load_data/csv
#https://www.tensorflow.org/tutorials/keras/save_and_load
def load_data_and_model():
    dataset = pd.read_csv("../Data/dataset_unrated_encoded_with_predictions.csv")
    column_names = [col for col in dataset.columns if col not in ['Title', 'Rating']]
    
    model = tf.keras.models.load_model("../Data/modell.keras")
    return dataset, column_names, model

#https://marcotcr.github.io/lime/tutorials/Using%2Blime%2Bfor%2Bregression.html
#https://lime-ml.readthedocs.io/en/latest/lime.html#module-lime.lime_tabular
#https://medium.com/data-science-in-your-pocket/lime-for-interpreting-machine-learning-models-maths-explained-with-codes-3d4226819020
def prepare_lime_explainer(X, column_names):
    explainer = LimeTabularExplainer(
        training_data=X,
        feature_names=column_names,
        mode='regression'
    )
    return explainer

#https://marcotcr.github.io/lime/tutorials/Using%2Blime%2Bfor%2Bregression.html
#https://stackoverflow.com/questions/9622163/save-plot-to-image-file-instead-of-displaying-it
#https://stackoverflow.com/questions/69924881/userwarning-starting-a-matplotlib-gui-outside-of-the-main-thread-will-likely-fa
def explain_top_recommendations(top_recommendations, dataset, explainer, model, column_names):
    results = []
    numberOfReccomendations = 3
    numberOfShownFeatures = 5
    for i in range(numberOfReccomendations):
        top_film_index = top_recommendations.index[i]
        top_film_features = dataset.iloc[top_film_index][column_names].values.reshape(1, -1)

        exp = explainer.explain_instance(
            data_row=top_film_features[0],
            predict_fn=lambda x: model.predict(x).flatten(),
            num_features=numberOfShownFeatures
        )
        
        explanation_list = exp.as_list()

        for feature, importance in explanation_list:
            print(f"Feature: {feature}, Importance: {importance}")
            
            results.append({
                "index": i,
                "feature": feature,
                "importance": importance
            })
    
        results_dataset = pd.DataFrame(results, columns=["index","feature", "importance"])
        results_dataset.to_csv("../Results/limeValues.csv", index=False, encoding="utf-8")

        fig = exp.as_pyplot_figure()
        plt.title(f'"{top_recommendations.iloc[i]["Title"]}"')
        plt.tight_layout()
    
        plt.savefig(f'../Results/lime_explanation_{i + 1}.png')
        plt.close(fig)

def do_lime_explanation():
    dataset, column_names, model = load_data_and_model()
    X = dataset[column_names].values
    explainer = prepare_lime_explainer(X, column_names)
    top_recommendations = pd.read_csv("../Results/top_recommendations.csv")
    explain_top_recommendations(top_recommendations, dataset, explainer, model, column_names)

def main():
    do_lime_explanation()

if __name__ == "__main__":
    main()
