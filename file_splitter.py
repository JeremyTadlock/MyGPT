def split_file(original_file, split_ratio=0.9):
    # Open the original file in read mode
    with open(original_file, 'r', encoding='utf-8') as file:
        # Read all lines from the original file
        lines = file.readlines()

    # Calculate the number of lines that corresponds to the split ratio
    split_point = int(len(lines) * split_ratio)

    # Define the names of the new files
    part1_file = f"train_{original_file}"
    part2_file = f"valid_{original_file}"

    # Write the first part of the lines to the first new file
    with open(part1_file, 'w', encoding='utf-8') as file:
        file.writelines(lines[:split_point])

    # Write the remaining lines to the second new file
    with open(part2_file, 'w', encoding='utf-8') as file:
        file.writelines(lines[split_point:])

    print(f"Split {original_file} into {part1_file} and {part2_file}")


# Example usage
original_file = 'input_data_files/TinyStoriesV2-GPT4-valid.txt'
split_file(original_file)