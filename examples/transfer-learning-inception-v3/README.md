```
#Donwload kaggle cats-vs-dogs dataset
kaggle competitions download -c dogs-vs-cats

# Set up the environment
python3 -m venv _env && source _env/bin/activate
pip install -r requirements.txt

# Launch Jupyter within the venv
pip3 install jupyter
jupyter notebook
```
