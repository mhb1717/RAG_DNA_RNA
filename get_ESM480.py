import os
import torch
import pathlib
import argparse
import numpy as np
from esm import pretrained

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-in", "--path_input", type=str, help="Path to input fasta folder")
parser.add_argument("-out", "--path_output", type=str, help="Path to output folder for .esm embeddings")

def main(input_folder, output_folder, miss_txt="esm_miss.txt", model_name="esm2_t12_35M_UR50D", batch_size=4):
    os.makedirs(output_folder, exist_ok=True)

    # Load ESM-2 model
    model, alphabet = pretrained.load_model_and_alphabet(model_name)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda().half()
        print("[INFO] Using GPU with float16")
    else:
        print("[INFO] Using CPU")

    batch_converter = alphabet.get_batch_converter()
    input_folder = pathlib.Path(input_folder)
    fasta_files = list(input_folder.glob("*.fasta"))

    # Read and prepare sequences
    sequences = []
    for fasta_file in fasta_files:
        try:
            with open(fasta_file) as f:
                lines = f.readlines()
                header = lines[0].strip().lstrip(">")
                sequence = lines[1].strip()
                sequences.append((header, sequence))
        except Exception as e:
            with open(miss_txt, 'a') as logf:
                logf.write(f"{fasta_file.name}: {e}\n")
            continue

    # Process in batches
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        try:
            labels, strs, tokens = batch_converter(batch)
            if torch.cuda.is_available():
                tokens = tokens.cuda()

            with torch.no_grad():
                out = model(tokens, repr_layers=[model.num_layers], return_contacts=False)

            reps = out["representations"][model.num_layers]

            for j, (label, seq) in enumerate(zip(labels, strs)):
                emb = reps[j, 1:len(seq)+1].cpu().numpy()  # remove [CLS] and [EOS]
                out_path = os.path.join(output_folder, f"{label}.esm")
                with open(out_path, 'w') as f_out:
                    for vec in emb:
                        line = ' '.join(map(str, vec))
                        f_out.write(line + '\n')

        except Exception as e:
            with open(miss_txt, 'a') as logf:
                logf.write(f"Batch {i}-{i+len(batch)} failed: {e}\n")
            print(f"[ERROR] Batch {i}-{i+len(batch)} failed: {e}")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args.path_input, args.path_output)
