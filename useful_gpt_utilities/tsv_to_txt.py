# Define the path to your TSV file and the output TXT file
tsv_file_path = 'input_data_files/Shellcode_IA32.tsv'
txt_file_path = 'input_data_files/Shellcode_IA32.txt'

# Open the TSV file and read its contents, ensuring UTF-8 encoding
with open(tsv_file_path, 'r', encoding='utf-8', errors='replace') as tsv_file:
    # Skip the header row
    next(tsv_file)
    # Read all lines from the TSV file and process them
    for line in tsv_file:
        # Split each line by tab to separate columns
        columns = line.strip().split('\t')
        # Reverse the order of columns and add "EOS" token
        # Ensure there are at least 2 columns to prevent errors
        if len(columns) >= 2:
            output_line = f'<question> {columns[1]}. <answer> {columns[0]} <eot>\n'

            # Open the TXT file in append mode to add the processed line, ensuring UTF-8 encoding
            with open(txt_file_path, 'a', encoding='utf-8', ) as txt_file:
                txt_file.write(output_line)
        else:
            print("ERROR, LESS THAN 2 COLUMNS: ", len(columns))
            for i, column in enumerate(columns):
                print(columns[i])
