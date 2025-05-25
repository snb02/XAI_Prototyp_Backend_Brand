from fastapi import FastAPI
from pydantic import BaseModel
import csv
import pandas as pd
import base64
#from typing import Optional

from AIRecommendationSystemComponent import do_ml_process
from LimeExplanationsComponent import do_lime_explanation
from CounterfactualExplanationsComponent import do_dice_explanation

#https://fastapi.tiangolo.com/de/
app = FastAPI()

#https://fastapi.tiangolo.com/de/tutorial/response-model/
class Rating(BaseModel):
    movie: str
    ratingNumber: float

#https://docs.python.org/3/library/csv.html
@app.post("/ratings")
def save_rating(rating: Rating):
    dataFile = "../Data/UserDataRating.csv"
    newline = []
    isMovieFound = False
    
    with open(dataFile, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row["Title"] == rating.movie:
                row["Rating"] = str(rating.ratingNumber)
                isMovieFound = True
            newline.append(row)

    if  isMovieFound:
        with open(dataFile, 'w', newline='') as csvfile:
            fieldnames = newline[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(newline)

    return {"Succes: Rating is saved"}

#https://stackoverflow.com/questions/71203579/how-to-return-a-csv-file-pandas-dataframe-in-json-format-using-fastapi
#https://medium.com/@liamwr17/supercharge-your-apis-with-csv-and-excel-exports-fastapi-pandas-a371b2c8f030
@app.get("/top3")
def get_top3Recommendations():
    do_ml_process()
    dataTopRecommendation = pd.read_csv("../Results/top_recommendations.csv")
    filme = dataTopRecommendation[['Title', 'predicted_rating']].to_dict(orient='records')

    return {"movieRecommendation": filme}
    
@app.get("/historyValues")
#https://stackoverflow.com/questions/71203579/how-to-return-a-csv-file-pandas-dataframe-in-json-format-using-fastapi
#https://medium.com/@liamwr17/supercharge-your-apis-with-csv-and-excel-exports-fastapi-pandas-a371b2c8f030
def get_history_Values():
    datahistoryValues = pd.read_csv("../Results/historyValues.csv")
    historyValues = datahistoryValues[['name', 'values']].to_dict(orient='records')

    return {"historyValues": historyValues}

#https://stackoverflow.com/questions/70710874/how-to-send-base64-image-using-python-requests-and-fastapi
@app.get("/lossImage")
def get_history_Image():
    with open("../Results/loss.png", "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode("utf-8")
            
    return {"lossImage": encoded}

#https://stackoverflow.com/questions/70710874/how-to-send-base64-image-using-python-requests-and-fastapi
class LimeImage(BaseModel):
    index: int
#    base64_png: Optional[str]
    image: str

@app.get("/limeImages")
def get_lime_explanations():
    do_lime_explanation()
    image_paths = [f"../Results/lime_explanation_{i}.png" for i in range(1, 4)]
    encoded_images = []
    
    for i, path in enumerate(image_paths):
        with open(path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode("utf-8")
            encoded_images.append(LimeImage(index=i, image=encoded))

    return {"lime_images": encoded_images}
 
@app.get("/limeValues")
#https://stackoverflow.com/questions/71203579/how-to-return-a-csv-file-pandas-dataframe-in-json-format-using-fastapi
#https://medium.com/@liamwr17/supercharge-your-apis-with-csv-and-excel-exports-fastapi-pandas-a371b2c8f030
def get_lime_Values():
    datalimeValues = pd.read_csv("../Results/limeValues.csv")
    limeValues = datalimeValues[['index','feature','importance']].to_dict(orient='records')

    return {"limeValues": limeValues}
    
#https://stackoverflow.com/questions/71203579/how-to-return-a-csv-file-pandas-dataframe-in-json-format-using-fastapi
#https://medium.com/@liamwr17/supercharge-your-apis-with-csv-and-excel-exports-fastapi-pandas-a371b2c8f030
@app.get("/counterfactualTexts")
def get_contractual_explanations():
    do_dice_explanation()
    dataCounterfactual = pd.read_csv("../Results/counterfactualExplanations.csv")
    counterfactualExplanations = dataCounterfactual[['title', 'explanations']].to_dict(orient='records')

    return {"counterfactualExplanations": counterfactualExplanations}
        
    
        
