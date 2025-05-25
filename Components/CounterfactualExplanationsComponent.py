import dice_ml
import pandas as pd
import tensorflow as tf
import numpy as np

#https://docs.seldon.io/projects/alibi/en/latest/overview/faq.html#im-getting-an-error-using-the-methods-counterfactual-counterfactualproto-or-cem-especially-if-trying-to-use-one-of-these-methods-together-with-integratedgradients-or-cfrl

#https://medium.com/@bijil.subhash/explainable-ai-diverse-counterfactual-explanations-dice-315f058c0364
#https://www.tensorflow.org/tutorials/load_data/csv
#https://www.tensorflow.org/tutorials/keras/save_and_load
def load_data_and_model():
    dataset = pd.read_csv("../Data/dataset_unrated_encoded_with_predictions.csv")
    column_names = [col for col in dataset.columns if col not in ['Title', 'Rating']]

    model = tf.keras.models.load_model("../Data/modell.keras")
    return dataset, column_names, model

#https://github.com/interpretml/DiCE
#https://interpret.ml/DiCE/notebooks/DiCE_getting_started.html#Explaining-a-Tensorflow-model
def prepare_dice_explainer(X, y, model, column_names):
    d = dice_ml.Data(
        dataframe=pd.concat([X, y.rename('Rating')], axis=1),
        continuous_features=[col for col in column_names if not any([col.startswith('Genre_'), col.startswith('Typ_'), col.startswith('Topic_')])],
        outcome_name='Rating'
    )

    m = dice_ml.Model(model=model, backend="TF2", model_type='regressor')
    exp = dice_ml.Dice(d, m, method="random")
    
    return exp

#https://github.com/interpretml/DiCE
#https://interpret.ml/DiCE/notebooks/DiCE_getting_started.html#Explaining-a-Tensorflow-model
#https://gegenfeld.com/docs/python/python-strings/verketten/
#https://codegree.de/python-replace/#:~:text=Verwendung%20der%20Python%20replace%20Methode,hello%20%3D%20'Hello%20world!'
def explain_top_recommendations(top_recommendations, dataset, exp, model, column_names, X):
    results = []
    rangeTitle = ""

    for index, row in top_recommendations.iterrows():
        title = row['Title']
        rating = row['predicted_rating']

        query_instance = X[dataset['Title'] == title]
        nuberOfCounterfactualExplanations = 1
        

        if query_instance.empty:
            print(f"Film '{title}' nicht in den Features gefunden.")
            continue
            
        searchedRatingRange = []
        if rating >= 7.0:
            searchedRatingRange = [1.0, 7.0]
            rangeTitle = "unter"
        else:
            searchedRatingRange = [7.0, 10.0]
            rangeTitle = "über"

        try:
            dice_exp = exp.generate_counterfactuals(
                query_instance,
                total_CFs=nuberOfCounterfactualExplanations,
                desired_range=searchedRatingRange,
                features_to_vary='all'
            )

            cf_dataset = dice_exp.cf_examples_list[0].final_cfs_df
            print(cf_dataset.T)
            
            original = query_instance.iloc[0]
            for idx, cf in cf_dataset.iterrows():
                satzbausteine = []
                for col in column_names:
                    orig_val = original[col]
                    cf_val = cf[col]
                    if orig_val != cf_val:
                        if "Genre_" in col and orig_val == 1 and cf_val == 0:
                            satzbausteine.append(f"wenn das Genre zu '{col.replace('Genre_', '')}' entfernt worden wäre")
                        elif "Genre_" in col and orig_val == 0 and cf_val == 1:
                            satzbausteine.append(f"wenn das Genre '{col.replace('Genre_', '')}' hinzugefügt worden wäre")
                        elif "Typ_" in col and orig_val == 1 and cf_val == 0:
                            satzbausteine.append(f"wenn der Typ '{col.replace('Typ_', '')}' entfernt worden wäre")
                        elif "Typ_" in col and orig_val == 0 and cf_val == 1:
                            satzbausteine.append(f"wenn der Typ '{col.replace('Typ_', '')}' hinzugefügt worden wäre")
                        elif "Themenstichwort_" in col and orig_val == 1 and cf_val == 0:
                            satzbausteine.append(f"wenn das Themenstichwort '{col.replace('Themenstichwort_', '')}' entfernt worden wäre")
                        elif "Themenstichwort_" in col and orig_val == 0 and cf_val == 1:
                            satzbausteine.append(f"wenn das Themenstichwort '{col.replace('Themenstichwort_', '')}' hinzugefügt worden wäre")
                        elif "Erscheinungsjahr" in col:
                            satzbausteine.append(f"wenn das Erscheinungsjahr von {int(orig_val)} auf {int(cf_val)} geändert worden wäre")
                if satzbausteine:
                    satz = " und ".join(satzbausteine)
                    explanation = f"Die KI hätte für {title} ein Rating von {rangeTitle} 7.0 vorhergesagt, {satz}."
                    print(explanation)
                    results.append([title, explanation])
                else:
                    explanation = f"Es wurden keine sinnvollen Möglichkeiten für {title} gefunden, um das Rating zu verbessern oder zu verschlechtern."
                    print(explanation)
                    results.append([title, explanation])
                
        except Exception as e:
            print(f"Es wurden keine sinnvollen Möglichkeiten für {title} gefunden, um das Rating zu verbessern oder zu verschlechtern.")
            explanation = f"Es wurden keine sinnvollen Möglichkeiten für {title} gefunden, um das Rating zu verbessern oder zu verschlechtern."
            results.append([title, explanation])

    results_dataset = pd.DataFrame(results, columns=["title", "explanations"])
    results_dataset.to_csv("../Results/counterfactualExplanations.csv", index=False, encoding="utf-8")

def do_dice_explanation():
    dataset, column_names, model = load_data_and_model()
    X = dataset[column_names].astype(np.float32)
    y = dataset['Rating'].astype(np.float32)
    explainer = prepare_dice_explainer(X, y, model, column_names)
    top_recommendations = pd.read_csv("../Results/top_recommendations.csv")
    explain_top_recommendations(top_recommendations, dataset, explainer, model, column_names, X)

def main():
    do_dice_explanation()

if __name__ == "__main__":
    main()
