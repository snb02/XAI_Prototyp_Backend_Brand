Titel: BA-PROTOTYP-XAI-BACKEND

Ersteller: Sophie Brand (5194731)

Prüfer: Thorsten Teschke

Beschreibung:  Dieser Ordner enthält das Backend des XAI-Prototyps der Bachelorarbeit „Erklärbare KI in mobilen Apps: Verbesserung der Transparenz und Verständlichkeit von KI-Systemen“.

Voraussetzung: Python https://www.python.org/ 
		Version: min. 3.8 max.3.11.0

Benötigte Python-Bibliotheken:

	pip install matplotlib

 	pip install lime

  	pip install pandas==1.5.3

    pip install numpy==1.26.4
 
	pip install "tensorflow>=2.16.1,<=2.19.0"
 
	pip install scikit-learn
 
	pip install fastapi
 
	pip install dice-ml
 
	pip install uvicorn
 
	pip install pydantic

 	Meine verwendeten Versionen (Python : 3.9.7, Tensorflow : 2.16.1, Numpy = 1.26.4, Pandas = 1.5.3)

Ausführung: 

	1. Terminal starten 
 
	2. Zum Ordner "Components" in XAI_Prototyp_Backend_Brand navigieren

	3. Server starten: 
		Entweder Lokal (bevorzugt): 
			uvicorn ServerConnectionComponent:app --reload
   
		Oder (Über WLAN für Testgeräte): 
			uvicorn ServerConnectionComponent:app --host 0.0.0.0 --port 8000

	Optional: 
 
		Einzelne Dateien starten: python3 Dateiname.py

Aufbau: 

Das Backend ist in mehrere Komponenten unterteilt: AIRecommendationSystemComponent, 																						CounterfactualExplanationsComponent,
						LimeExplanationsComponent,
						ServerConnectionComponent,
						UserDataRating (Datensatz),
						MovieData (Datensatz),
						Weitere CSV Datein & Bilder (Die Resultate der Komponenten)

Hinweise: Dies stellt lediglich das Backend dar und ist für den Einsatz mit einer begleitenden iOS-App konzipiert.
	
