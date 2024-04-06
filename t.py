# Step 1: Open the original file
with open('TinyStoriesV2-GPT4-train.txt', 'r', encoding='utf-8') as file:
    content = file.read()  # Step 2: Read content

# Step 3: Calculate the length of the first 50%
half_length = len(content) // 2

# Step 4: Open a new file
with open('50_TinyStoriesV2-GPT4-train.txt', 'w', encoding='utf-8') as new_file:
    # Step 5: Write the first 50% to the new file
    new_file.write(content[:half_length])