import os
import random
import shutil
from collections import defaultdict

def get_sequence_groups(files):
    sequence_groups = []
    
    # sort after sequence number per original file
    sorted_files = sorted(files, key=lambda x: int(os.path.basename(x).split('_')[1]))
    current_group = [sorted_files[0]]
    
    # bundle continious sequences in groups
    for i in range(1, len(sorted_files)):
        if int(os.path.basename(sorted_files[i]).split('_')[1]) == int(os.path.basename(sorted_files[i - 1]).split('_')[1]) + 1:
            current_group.append(sorted_files[i])
        else:
            sequence_groups.append(current_group)
            current_group = [sorted_files[i]]
    sequence_groups.append(current_group)
    return sequence_groups

def group_files_by_sequence(folder):
    sequence_files = defaultdict(list)
    for root, dirs, files in os.walk(folder):
        for file_name in files:
            parts = os.path.basename(file_name).split('_')
            if len(parts) >= 3:
                name = '_'.join(parts[2:]).rsplit('.', 1)[0]
                sequence_files[name].append(os.path.join(root, file_name))

    sequence_groups = {}
    for key, files in sequence_files.items():
        sequence_groups[key] = get_sequence_groups(files)
    return sequence_groups

def get_files_without_sequence(folder):
    collected_files = []
    for root, dirs, files in os.walk(folder):
        for file_name in files:
            collected_files.append(os.path.join(root, file_name))
    return collected_files

def split_data(data, group_by_sequence=False, ratio=[0.8, 0.1, 0.1]):
    assert sum(ratio) == 1, "Ratio values must add up to 1"
    if group_by_sequence:
        sequences = [item for sublist in data.values() for item in sublist]
        
        #---Infoprint
        print("First entry of sequences before shuffle:", sequences[0])
        #---
        
        random.shuffle(sequences)
        
        #---Infoprint
        first_data_key = next(iter(data.keys()))
        print("First entry of data.keys():", first_data_key)
        print("First entry of sequences after shuffle:", sequences[0])
        #---
        
        n = len(sequences)
        n_train = int(n * ratio[0])
        n_val = int(n * ratio[1])

        train_sequences = sequences[:n_train]
        val_sequences = sequences[n_train:n_train+n_val]
        test_sequences = sequences[n_train+n_val:]

        train = [file_name for sequence in train_sequences for file_name in sequence]
        val = [file_name for sequence in val_sequences for file_name in sequence]
        test = [file_name for sequence in test_sequences for file_name in sequence]
        
        #---Infoprint
        print("First 5 entries of train:", train[:5])
        print("First 5 entries of val:", val[:5])
        print("First 5 entries of test:", test[:5])
        #---
        
        return train, val, test
    else:
        random.shuffle(data)
        n = len(data)
        n_train = int(n * ratio[0])
        n_val = int(n * ratio[1])

        train = data[:n_train]
        val = data[n_train:n_train+n_val]
        test = data[n_train+n_val:]
        return train, val, test

def copy_files_to_destination(files, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    for file_path in files:
        destination = os.path.join(destination_folder, os.path.basename(file_path))
        shutil.copy2(file_path, destination)

def main():
    folders_with_sequence = ["/home/cut-data/source/corn_bunting/", "/home/cut-data/source/other_birds/"]
    folder_without_sequence = "/home/cut-data/source/miscellaneous/audioset/No-Target"
    base_destinations = "/home/cut-data"

    for folder in folders_with_sequence:
        sequence_files = group_files_by_sequence(folder)
        
        #---DEBUG
        first_entry_key = next(iter(sequence_files))
        first_entry_value = sequence_files[first_entry_key]
        print(f"In {folder}, the first full entry of sequence_files is: {first_entry_key} - {first_entry_value}")
        #---

        train_seq, val_seq, test_seq = split_data(sequence_files, group_by_sequence=True)

        # Debug: Print the number of files in each dataset split
        print(f"Training files: {len(train_seq)}, validation files: {len(val_seq)}, test files: {len(test_seq)}")

        destinations = [os.path.join(base_destinations, subfolder) for subfolder in ["training", "validation", "testing"]]

        target_folder_name = "target" if folder == "/home/cut-data/source/corn_bunting/" else "non_target"
        
        for files, destination in zip([train_seq, val_seq, test_seq], destinations):
            dest_path = os.path.join(destination, target_folder_name)
            print(f"Starting filling {len(files)} files of {folder} folder in {dest_path}")
            copy_files_to_destination(files, dest_path)
            print(f"Stopping filling with {len(os.listdir(dest_path))} files in {dest_path}")

    no_sequence_files = get_files_without_sequence(folder_without_sequence)
    print(f"Number of no sequence files: {len(no_sequence_files)}") #Debug
    train_no_seq, val_no_seq, test_no_seq = split_data(no_sequence_files)

    destinations = [os.path.join(base_destinations, subfolder) for subfolder in ["training", "validation", "testing"]]

    for files, destination in zip([train_no_seq, val_no_seq, test_no_seq], destinations):
        dest_path = os.path.join(destination, "non_target")
        print(f"Starting filling {len(files)} files of {folder_without_sequence} folder in {dest_path}")
        copy_files_to_destination(files, dest_path)
        print(f"Stopping filling after {len(os.listdir(dest_path))} files")

if __name__ == "__main__":
    main()