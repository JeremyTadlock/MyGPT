import os


def split_file(original_file, split_ratio=0.85):
    # Extract directory and filename from the original file path
    dir_name, filename = os.path.split(original_file)

    # Calculate the new filenames, keeping them in the same directory
    part1_file = os.path.join(dir_name, f"train_{filename}")
    part2_file = os.path.join(dir_name, f"valid_{filename}")

    # Open the original file in read mode
    with open(original_file, 'r', encoding='utf-8') as file:
        # Read all lines from the original file
        lines = file.readlines()

    # Calculate the split point
    split_point = int(len(lines) * split_ratio)

    # Write the first part to the train file
    with open(part1_file, 'w', encoding='utf-8') as file:
        file.writelines(lines[:split_point])

    # Write the second part to the valid file
    with open(part2_file, 'w', encoding='utf-8') as file:
        file.writelines(lines[split_point:])

    print(f"Split {original_file} into {part1_file} and {part2_file}")


# Example usage
original_file = '../input_data_files/50percent_TinyStoriesV2-GPT4-train.txt'
split_file(original_file)
