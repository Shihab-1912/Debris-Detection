# Space Debris Risk Detection

Streamlit deployment package for the Space Debris Detection notebook.

## Folder structure

```text
space_debris_streamlit/
+-- app.py
+-- app/
|   +-- __init__.py
|   +-- model_utils.py
+-- data/
|   +-- space_debris_uncleaned.csv
+-- models/
|   +-- space_debris_model.joblib
+-- scripts/
|   +-- train_model.py
+-- .streamlit/
|   +-- config.toml
+-- .gitignore
+-- README.md
+-- requirements.txt
```

## Run locally

```bash
cd space_debris_streamlit
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/train_model.py
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Deploy on Streamlit Community Cloud

1. Create a new GitHub repository.
2. Upload or push the contents of this `space_debris_streamlit` folder.
3. Go to [Streamlit Community Cloud](https://share.streamlit.io/).
4. Click **New app**.
5. Select your repository, branch, and set the main file path to `app.py`.
6. Click **Deploy**.

The saved model is already included in `models/space_debris_model.joblib`. It is compressed so it stays under GitHub's normal file-size limit. If you update the dataset later, rerun:

```bash
python scripts/train_model.py
```

Then commit the updated model file.

## GitHub commands

From inside `space_debris_streamlit`:

```bash
git init
git add .
git commit -m "Add Streamlit space debris risk app"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

Replace `YOUR_USERNAME` and `YOUR_REPOSITORY` with your GitHub details.

## What the app does

- Cleans the original space debris dataset.
- Engineers orbital risk features used in the notebook.
- Trains a Random Forest classifier.
- Lets users enter debris/orbital measurements and predict `Low`, `Medium`, or `High` risk.
- Includes charts for altitude, debris type, and risk distribution.
