Quick Start
Step 1: Generate Data Features
Navigate to the 'data' folder and use the FASTA file to generate additional data features that are saved in the 'dataset' folder..

Example usage:

python get_ProtTrans.py -in "Your FASTA file folder" -out "The destination folder of your output"
python get_tape.py -in "Your FASTA file folder" -out "The destination folder of your output"
python get_esm.py "Pretrained model of ESM" "Your FASTA file folder" "The destination folder of your output" --repr_layers 33 --include per_tok
python get_ProstT5.py -in "Your FASTA file folder" -out "The destination folder of your output"
Step 2: Create a multi-scale or the sliding windows feature set.
Run get_dataset.py:

Python get_dataset.py -in "path of features" -label "path of labels" -out "path of output features set" -w 7 -dt ".prottrans"
w is sliding window size; in our case, we use 7
The feature type: use 'dt' for prottrans, 'esm' for ESM, and 'tape' for TAPE, etc
Set paths in file import_test.py:

train_data.npy: Contains the training data.
train_labels.npy: Contains the corresponding training labels.
testing_data.npy: Contains the testing data.
testing_labels.npy: Contains the corresponding testing labels.
Step 3: RAG Strategy
Create RAG-DB:
Run the get_dataset_RAG_DB.npy file to creat RAG-DB or External DataBase.
Run the get_RagEmb_Batch.npy file to create the final embeddings for each training and testing..
Step 4: Execute Prediction
Run the Model:
Open the DeepAtten_Metal.ipynb file in Jupyter Notebook.
Execute the cells in the notebook to run the model and make predictions based on your Final RAG-Embeddings.
