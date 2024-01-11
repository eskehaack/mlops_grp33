import json
import os


def txt_to_json(txt_file_path):
    # Check if the file exists
    if not os.path.exists(txt_file_path):
        print(f"The file {txt_file_path} does not exist.")
        return

    # Extract the directory and filename without extension
    directory, filename = os.path.split(txt_file_path)
    filename_without_ext = os.path.splitext(filename)[0]

    # Read the .txt file
    with open(txt_file_path, "r") as file:
        lines = [line.strip() for line in file.readlines()]

    # Convert to JSON
    json_data = json.dumps(lines, indent=4)

    # Save as .json in the same location
    json_file_path = os.path.join(directory, filename_without_ext + ".json")
    with open(json_file_path, "w") as file:
        file.write(json_data)

    print(f"File saved as {json_file_path}")


# Example usage
txt_to_json("data/label_mapping.txt")
