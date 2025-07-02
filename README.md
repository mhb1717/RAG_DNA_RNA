**Example usage:**
```bash
python get_ProtTrans.py ...
python get_ProstT5.py ...
```  ← ✅ THIS closes the Step 1 code block
## Quick Start

### Step 1: Generate Data Features
Navigate to the `data` folder and use the **FASTA** file to generate additional data features saved in the `dataset` folder.

**Example usage:**
```bash
python get_ProtTrans.py -in "Your FASTA file folder" -out "The destination folder of your output"
python get_tape.py -in "Your FASTA file folder" -out "The destination folder of your output"
python get_esm.py "Pretrained model of ESM" "Your FASTA file folder" "The destination folder of your output" --repr_layers 33 --include per_tok
python get_ProstT5.py -in "Your FASTA file folder" -out "The destination folder of your output"
