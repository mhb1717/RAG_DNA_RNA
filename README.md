🚀 Quick Start
Step 1: Generate Data Features
Navigate to the data folder and use the FASTA file to generate additional data features saved in the dataset folder.

Example usage:

bash
Copy
Edit
python get_ProtTrans.py -in "Your FASTA file folder" -out "The destination folder of your output"
python get_tape.py -in "Your FASTA file folder" -out "The destination folder of your output"
python get_esm.py "Pretrained model of ESM" "Your FASTA file folder" "The destination folder of your output" --repr_layers 33 --include per_tok
python get_ProstT5.py -in "Your FASTA file folder" -out "The destination folder of your output"
Step 2: Create Multi-Scale or Sliding Window Feature Set
Run get_dataset.py:

bash
Copy
Edit
python get_dataset.py -in "path of features" -label "path of labels" -out "path of output features set" -w 7 -dt ".prottrans"
w: sliding window size (e.g., 7)

dt: feature type — use .prottrans, .esm, .tape, etc.

Set paths in import_test.py:

train_data.npy: Training data

train_labels.npy: Training labels

testing_data.npy: Testing data

testing_labels.npy: Testing labels

Step 3: RAG Strategy
Create RAG-DB:

Run get_dataset_RAG_DB.npy to create the RAG-DB or external database.

Run get_RagEmb_Batch.npy to create final embeddings for training and testing.

Step 4: Execute Prediction
Run the Model:

Open DeepAtten_Metal.ipynb in Jupyter Notebook.

Execute all cells to run the model and make predictions using your final RAG embeddings.

