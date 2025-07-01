import os
import re
import torch
import numpy as np
import argparse
from transformers import T5Tokenizer, T5EncoderModel
from Bio import SeqIO
from tqdm import tqdm

# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load ProstT5 model
print("Loading ProstT5 model...")
tokenizer = T5Tokenizer.from_pretrained('Rostlab/ProstT5', do_lower_case=False)
model = T5EncoderModel.from_pretrained("Rostlab/ProstT5").to(device)
model.eval()
model = model.half() if device.type != 'cpu' else model.float()

# Function to clean and prepare protein sequences
def prepare_sequence(sequence):
    cleaned_sequence = re.sub(r"[UZOB]", "X", sequence)
    return "<AA2fold> " + " ".join(list(cleaned_sequence))

# Function to extract embeddings from a sequence
def extract_embeddings(sequence):
    ids = tokenizer.batch_encode_plus(
        [sequence],
        add_special_tokens=True,
        padding="longest",
        return_tensors='pt'
    ).to(device)
    with torch.no_grad():
        embedding_output = model(
            ids.input_ids,
            attention_mask=ids.attention_mask
        )
    embeddings = embedding_output.last_hidden_state[0, 1:len(sequence.split()) + 1]
    return embeddings.cpu()

# Main function to process input and output
def main(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    print("Processing FASTA files and extracting embeddings...")

    for fasta_file in tqdm(os.listdir(input_folder), desc="Files Processed"):
        if fasta_file.endswith(".fasta") or fasta_file.endswith(".fa"):
            input_path = os.path.join(input_folder, fasta_file)
            output_path = os.path.join(output_folder, os.path.splitext(fasta_file)[0] + ".pt")

            all_embeddings = []
            for record in SeqIO.parse(input_path, "fasta"):
                print(f"Processing sequence: {record.id}")
                cleaned_sequence = prepare_sequence(str(record.seq))
                embeddings = extract_embeddings(cleaned_sequence)
                all_embeddings.append(embeddings)

            # Save all embeddings as a list of tensors
            torch.save(all_embeddings, output_path)
            print(f"Saved embeddings to: {output_path}")

    print("All embeddings have been extracted successfully!")

# Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract ProstT5 embeddings and save as .pt.")
    parser.add_argument("-in", "--path_input", type=str, required=True, help="Path to input folder with FASTA files.")
    parser.add_argument("-out", "--path_output", type=str, required=True, help="Path to save .pt embeddings.")
    args = parser.parse_args()
    main(args.path_input, args.path_output)
