from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch

# Dictionary to store models and tokenizers
model_store = {}

def get_modernbert_model(device: str):
    if "modernbert" not in model_store:
        print("Loading ModernBERT model...")
        tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
        model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")
        model.to(device)
        model.eval()
        model_store["modernbert"] = (tokenizer, model)
    return model_store["modernbert"]

def get_qwen_model(device: str):
    if "qwen" not in model_store:
        print("Loading Qwen model...")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B")
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-0.5B")
        model.to(device)
        model.eval()
        model_store["qwen"] = (tokenizer, model)
    return model_store["qwen"]

@torch.no_grad()
def forward_modernbert(input_text, device: str):
    tokenizer, model = get_modernbert_model(device)
    tokens = tokenizer(input_text, return_tensors="pt", truncation=True).to(device)
    tokens['input_ids'] = tokens['input_ids'].to(torch.long)
    output = model(**tokens)
    # Example: return the mean of the last hidden state
    breakpoint()
    return output.last_hidden_state.mean(dim=1)

@torch.no_grad()
def forward_modernbert_two_inputs(input_text1, input_text2, device: str):
    tokenizer, model = get_modernbert_model(device)
    
    # Tokenize both inputs at once to ensure they are padded to the same length.
    tokens = tokenizer(
        [input_text1, input_text2],
        padding=True,        # pads to the length of the longest sequence in the batch
        truncation=True,
        return_tensors="pt"
    ).to(device)
    
    # (Optional) Ensure the input_ids are in torch.long format.
    tokens['input_ids'] = tokens['input_ids'].to(torch.long)
    
    # Forward the padded inputs through the model.
    output = model(input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask'])

    return output.last_hidden_state


@torch.no_grad()
def forward_qwen(input_text, device: str):
    tokenizer, model = get_qwen_model(device)
    tokens = tokenizer(input_text, return_tensors="pt", truncation=True).to(device)
    tokens['input_ids'] = tokens['input_ids'].to(torch.long)
    output = model(**tokens)
    # Example: return the logits for the last token
    return output.logits[:, -1]

# Example usage
def get_embedding_with_modernbert(input_text, device: str):
    embedding = forward_modernbert(input_text, device)
    print("ModernBERT embedding:", embedding.shape)

def get_logits_with_qwen(input_text, device: str):
    logits = forward_qwen(input_text, device)
    print("Qwen logits:", logits.shape)
    
    
if __name__ == "__main__":
    # get_embedding_with_modernbert("def add(a, b): return a + b", "cpu")
    # # get_logits_with_qwen("def add(a, b): return a + b", "cpu")
    # get_embedding_with_modernbert("def add(a, b): return a + b", "cpu")
    
    string1 = '''def add(a, b):  return a + b'''
    string2 = '''def add(a, b):  return a + b'''
    # get_embedding_with_modernbert_two_inputs(string1, string2, "cpu")
    embedding1, embedding2 = forward_modernbert_two_inputs(string1, string2, "cpu")
    print(embedding1.shape)
    print(embedding2.shape)
    cosine_sim = torch.nn.functional.cosine_similarity(
        embedding1,  # Shape: (seq_length, 1, hidden_size)
        embedding2,  # Shape: (1, seq_length, hidden_size)
        dim=-1
    )
    print(cosine_sim.shape)
    print(cosine_sim.mean())
