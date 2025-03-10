import os
from utils.chunking_text import split_text_into_chunks


input_dir = "data/extracted_text/"
output_dir = "data/chunked_text/"


os.makedirs(output_dir, exist_ok=True)


for filename in os.listdir(input_dir):
    if filename.endswith(".txt"):
        file_path = os.path.join(input_dir, filename)
        
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = split_text_into_chunks(text, max_length=1000, overlap_length=200)

        chunked_filename = filename.replace(".txt", "_chunks.txt")
        output_path = os.path.join(output_dir, chunked_filename)
        
        with open(output_path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(chunk + "\n\n")  
        
        print(f"✅ Chunked text saved: {output_path}")
