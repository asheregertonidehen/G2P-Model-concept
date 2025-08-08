Clone the repository

Create a virtual environment: python -m venv venv

Activate it: source venv/bin/activate (Linux/Mac) or venv\Scripts\activate (Windows)

Install dependencies: pip install -r requirements.txt

(old model) Run the model: python [g2p_model_edo.py](http://_vscodecontentref_/0) predict 

(new model) Train model: edo_g2p_best.py train (can specify epochs)
(new model) Run model: edo_g2p_best.py predict

The new model includes several optimization features:

- Early Stopping: Will stop if validation doesn't improve for 8 epochs
  
- Learning Rate Scheduling: Reduces learning rate when validation loss plateaus
  
- Teacher Forcing: Gradually reduces from 100% to 30% over training
  
- Mixed Precision: Uses AMP for faster training on GPU
