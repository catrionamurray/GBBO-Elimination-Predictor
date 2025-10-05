# GBBO-Elimination-Predictor
A fun project exploring the Great British Bake Off (GBBO) results across Seasons 3-16! 🎂 The goal was to predict when a contestant will be eliminated based on their past weekly performance, using data from 14 seasons of GBBO (sourced from [Kaggle](https://www.kaggle.com/datasets/sarahvitvitskiy/great-british-bake-off-results-seasons-1-14) and Wikipedia).

Logistic regression model trained on Seasons 3-9 and tested on Seasons 10-14:
[Launched here!](https://gbbo-predictor.streamlit.app)

1) Includes per-season elimination prediction heatmaps (compared to real elimination and star bakers):

<img width="668" height="407" alt="Screenshot 2025-10-05 at 3 15 02 PM" src="https://github.com/user-attachments/assets/e5dee9eb-f404-4cd9-942e-da308fd18ff7" />


2) Includes logistic regression coefficients to explore feature importance:
<img width="694" height="367" alt="Screenshot 2025-10-05 at 3 16 44 PM" src="https://github.com/user-attachments/assets/87bec394-dc17-491b-bc03-b6f6e6255cc6" />


3) Includes super-preliminary win probabilities estimated from survival probabilities:
<img width="745" height="370" alt="Screenshot 2025-10-05 at 3 18 32 PM" src="https://github.com/user-attachments/assets/09be26e8-e3fc-41a5-b4a4-d89cbe1a495d" />
