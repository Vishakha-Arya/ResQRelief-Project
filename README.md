# ResQRelief-Project

ResQRelief is a machine learning project for disaster impact prediction and response management.  
The MVP starts with flood prediction, but the system is designed to extend to other disasters in the future.  

## Project Deliverables

-   **Final Source Code:** This repository contains all project files, including the Flask application.
-   **Trained Models:** The `.pkl` files for the trained models are available in this Google Drive folder:
    -   **Google Drive Link:** [https://drive.google.com/drive/folders/1aWfmy5DbpiCGwYhv_TzkRtvEKftOiv2n?usp=sharing](https://drive.google.com/drive/folders/1aWfmy5DbpiCGwYhv_TzkRtvEKftOiv2n?usp=sharing)

## Datasets
- `flood.csv` – flood-related data  
- `disaster_messages.csv` – disaster-related messages  
- `disaster_categories.csv` – categories linked to messages  
- `india-districts-census-2011.csv` – demographic and regional data  

### Week 1: Data Exploration
-   Imported required libraries
-   Loaded datasets into Colab
-   Performed initial checks: `.head()`, `.info()`, `.isnull().sum()`, `.describe()`

### Week 2: EDA & Data Transformation
-   **Data Transformation:** Merged message and category datasets and transformed the multi-label categories into separate binary columns.
-   **Exploratory Data Analysis (EDA):** Performed univariate and bivariate analysis on the flood and census data using various visualizations (histograms, boxplots, correlation heatmaps).
-   **Preprocessing:** Implemented feature scaling and handled categorical features to prepare the data for model training.

### Week 3: Model Building & Deployment (MVP)
-   **Model Building:** Trained a **Random Forest Classifier** for flood prediction and a **Multi-Output Classifier** for message classification.
-   **Model Persistence:** Saved the trained models as `.pkl` files to enable reusability and deployment.
-   **MVP Deployment:** Created a working `app.py` file using the **Flask** framework to demonstrate the project's functionality in a practical, user-friendly web application.

## License
MIT License.  
Datasets used in this project are open-source and credited to their respective providers (Kaggle / Census Bureau / Figure Eight). This repository is for educational purposes only.
