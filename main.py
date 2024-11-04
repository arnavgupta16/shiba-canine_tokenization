import PyPDF2
from shiba import Shiba, CodepointTokenizer, get_pretrained_from_hub

# Load SHIBA model and tokenizer
shiba_model = Shiba()
shiba_model.load_state_dict(get_pretrained_from_hub())
shiba_model.eval()  # Disable dropout
tokenizer = CodepointTokenizer()

def extract_text_from_pdf(pdf_path):
    """Extract text from a given PDF file."""
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def split_text(text, max_length=1800):
    """Split text into chunks within the max length allowed by SHIBA."""
    return [text[i:i + max_length] for i in range(0, len(text), max_length)]

def process_pdf_with_shiba(pdf_path):
    # Extract and clean text from PDF
    text = extract_text_from_pdf(pdf_path)
    # Split text into manageable chunks
    text_chunks = split_text(text)
    all_outputs = []
    encoded_chunks = []

    for i, chunk in enumerate(text_chunks):
        # Encode each chunk
        encoded = tokenizer.encode_batch([chunk])
        print(f"Chunk {i} encoding result:", encoded)  # Debugging statement

        # Access `input_ids` and `attention_mask` and structure inputs for the model
        if 'input_ids' in encoded and 'attention_mask' in encoded:
            input_ids = encoded['input_ids']
            attention_mask = encoded['attention_mask']
            inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
            outputs = shiba_model(**inputs)
            all_outputs.append(outputs)
            encoded_chunks.append(input_ids[0])  # Store input_ids for decoding

    return all_outputs, encoded_chunks

def decode_tokens(encoded_chunks):
    decoded_text = ""
    for tokens in encoded_chunks:
        decoded_text += tokenizer.decode(tokens)
    return decoded_text

def save_text_to_txt(text, output_txt_path):
    """Save the provided text to a .txt file."""
    with open(output_txt_path, "w", encoding="utf-8") as file:
        file.write(text)

# Example usage
pdf_path = "Speech-of-Barack-Obama.pdf"
output_txt_path = "Processed-Speech-of-Barack-Obama.txt"
outputs, encoded_chunks = process_pdf_with_shiba(pdf_path)
decoded_text = decode_tokens(encoded_chunks)
print("Decoded Text:", decoded_text)

# Save the decoded text as a .txt file
save_text_to_txt(decoded_text, output_txt_path)
print(f"Processed text saved as {output_txt_path}")
