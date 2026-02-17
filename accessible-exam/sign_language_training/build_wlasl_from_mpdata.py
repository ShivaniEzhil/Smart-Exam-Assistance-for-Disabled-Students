"""
Build WLASL_Hand_Data from existing MP_Data (alphabet sequences).
Use this when you cannot run extract_wlasl_hand_sequences (e.g. no WLASL videos
or MediaPipe fails). The "word" model will then recognize the same letters
(A-Z, 1-9) as the alphabet model — Record word will insert one letter per sign.

Run from accessible-exam:  python sign_language_training/build_wlasl_from_mpdata.py
"""
import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MP_DATA = os.path.join(PROJECT_ROOT, "MP_Data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "WLASL_Hand_Data")
SEQ_LEN = 30
FEAT_DIM = 63

def main():
    if not os.path.isdir(MP_DATA):
        print("MP_Data not found. Run process_images.py first.")
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    count = 0
    for cls in sorted(os.listdir(MP_DATA)):
        class_path = os.path.join(MP_DATA, cls)
        if not os.path.isdir(class_path):
            continue
        out_class = os.path.join(OUTPUT_DIR, cls)
        os.makedirs(out_class, exist_ok=True)
        for seq_id in sorted(os.listdir(class_path), key=lambda x: int(x) if x.isdigit() else 0):
            seq_path = os.path.join(class_path, seq_id)
            if not os.path.isdir(seq_path):
                continue
            frames = []
            for i in range(SEQ_LEN):
                f = os.path.join(seq_path, f"{i}.npy")
                if not os.path.isfile(f):
                    break
                arr = np.load(f)
                if arr.size != FEAT_DIM:
                    break
                frames.append(arr.flatten())
            if len(frames) == SEQ_LEN:
                out_arr = np.array(frames, dtype=np.float32)
                out_path = os.path.join(out_class, f"{seq_id}.npy")
                np.save(out_path, out_arr)
                count += 1
    print(f"Built WLASL_Hand_Data: {count} sequences. Run train_word_model.py next.")

if __name__ == "__main__":
    main()
