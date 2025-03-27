# Otitenet

Use python3.11

python -m pip install -r requirements.txt

python setup.py

make_dataset2.py

train_triplet_new.py

# MLFLOW

ssh -L 8502:localhost:8502 simon@198.168.189.19

streamlit run app.py --server.address 0.0.0.0 --server.port 8502

# make new db
python init_db.py

sudo mysql -u root -p
-- Create the database
CREATE DATABASE results_db;

-- Grant privileges to the user
GRANT ALL PRIVILEGES ON results_db.* TO 'y_user'@'%';

-- Set the user's password
ALTER USER 'y_user'@'%' IDENTIFIED BY 'password';

-- Apply the changes
FLUSH PRIVILEGES;
