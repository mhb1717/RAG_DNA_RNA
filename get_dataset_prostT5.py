import argparse
import numpy as np
import os
import torch
from sklearn.preprocessing import OneHotEncoder

# === Argument parser ===
parser = argparse.ArgumentParser()
parser.add_argument("-in", "--path_input", type=str, help="Path to input .pt files")
parser.add_argument("-label", "--label_path", type=str, help="Path to label .txt files")
parser.add_argument("-out", "--path_output", type=str, help="Path to output directory")
parser.add_argument("-w", "--window_size", type=int, help="Sliding window size")
parser.add_argument("-dt", "--data_type", type=str, default=".pt", help="Input file extension (default: .pt)")

# === Load PyTorch .pt feature ===
def loadData(path):
    data = torch.load(path, map_location='cpu')
    if isinstance(data, dict) and 'last_hidden_state' in data:
        data = data['last_hidden_state']
    arr = np.array(data)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]  # Remove batch dimension
    return arr.astype('float16')

# === Load label ===
def loadlabel(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        y_data = [int(x) for x in lines[1].strip()]
    return np.array(y_data).astype('float16')

# === One-hot encode labels ===
def one_hot_encode(labels):
    enc = OneHotEncoder(sparse_output=False)
    labels = labels.reshape(-1, 1)
    return enc.fit_transform(labels).astype('float16')

# === Apply sliding window ===
def get_series_feature(data, window_size):
    total_len = window_size * 2 + 1
    padded = np.pad(data, ((window_size, window_size), (0, 0)), mode='constant')
    result = np.zeros((data.shape[0], total_len, data.shape[1]), dtype='float16')
    for i in range(data.shape[0]):
        result[i] = padded[i:i + total_len]
    return result

# === Save .npy files ===
def saveData(output_dir, data, labels):
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Saving to: {output_dir}")
    print(f"[INFO] Data shape: {data.shape}")
    print(f"[INFO] Label shape: {labels.shape}")
    np.save(os.path.join(output_dir, "TR646_data_Pt.npy"), data)
    np.save(os.path.join(output_dir, "TR646_label_pt.npy"), labels)

# === Main pipeline ===
def main(path_input, path_output, window_size, data_type, label_path):
    feature_list = []
    label_list = []

    input_files = [f for f in os.listdir(path_input) if f.endswith(data_type)]
    print(f"[INFO] Found {len(input_files)} feature files.")

    for file in input_files:
        base = file.replace(data_type, "")  # e.g. 4zm2_B
        feat_file = os.path.join(path_input, file)
        label_file = os.path.join(label_path, f"{base}txt")

        if not os.path.exists(label_file):
            print(f"[WARNING] Missing label file: {label_file}, skipping.")
            continue

        try:
            data = loadData(feat_file)
            labels = loadlabel(label_file)

            # === Auto-trim if off by 1 ===
            if abs(data.shape[0] - len(labels)) == 1:
                min_len = min(data.shape[0], len(labels))
                data = data[:min_len]
                labels = labels[:min_len]
                print(f"[FIXED] Trimmed {file} to match lengths: {min_len}")
            elif data.shape[0] != len(labels):
                print(f"[WARNING] Mismatched lengths (data: {data.shape[0]}, labels: {len(labels)}) in {file}, skipping.")
                continue

            feature_list.append(get_series_feature(data, window_size))
            label_list.append(labels)

        except Exception as e:
            print(f"[ERROR] Failed to process {file}: {e}")
            continue

    if not feature_list or not label_list:
        print("[ERROR] No valid data-label pairs processed.")
        return

    X = np.expand_dims(np.concatenate(feature_list, axis=0), axis=1)
    Y = one_hot_encode(np.concatenate(label_list, axis=0))

    saveData(path_output, X, Y)
    print("[DONE] All processing completed successfully.")

# === Entry point ===
if __name__ == "__main__":
    args = parser.parse_args()
    main(
        path_input=args.path_input,
        path_output=args.path_output,
        window_size=args.window_size,
        data_type=args.data_type,
        label_path=args.label_path
    )
