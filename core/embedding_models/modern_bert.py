import torch
from core.embedding_models.embedding_models import get_modernbert_model

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
def forward_modernbert(input_text, device: str):
    tokenizer, model = get_modernbert_model(device)
    tokens = tokenizer(input_text, return_tensors="pt", truncation=True).to(device)
    # print(tokens)
    tokens['input_ids'] = tokens['input_ids'].to(torch.long)
    output = model(**tokens)
    # Example: return the mean of the last hidden state
    return output.last_hidden_state #.mean(dim=1)


def cosine_similarity_pooled(embedding1, embedding2):
    return torch.nn.functional.cosine_similarity(
        embedding1.mean(dim=0),  # Shape: (seq_length, hidden_size)
        embedding2.mean(dim=0),  # Shape: (seq_length, hidden_size)
        dim=-1
    )

def cosine_similarity_cls(embedding1, embedding2):
    return torch.nn.functional.cosine_similarity(
        embedding1[0],  # Shape: (seq_length, hidden_size)
        embedding2[0],  # Shape: (seq_length, hidden_size)
        dim=-1
    )

def cosine_similarity_dense(reference_embedding, prediction_embedding):
    
    sim_matrix = torch.nn.functional.cosine_similarity(
        reference_embedding.unsqueeze(1),  # Shape: (seq_length, 1, hidden_size)
        prediction_embedding.unsqueeze(0),  # Shape: (1, seq_length, hidden_size)
        dim=-1
    )
    
    most_similar_ref_scores, most_similar_ref_idx = sim_matrix.max(dim=0)
    # reference_embedding_most_similar = reference_embedding[most_similar_ref_idx]
    # breakpoint()
    return most_similar_ref_scores.mean()

if __name__ == "__main__":
    # string1 = '''def add(apple, banana):  return apple + banana'''
    string1 = '''@torch.no_grad()\ndef add(a, b, c):  return a + b + c'''
    string2 = '''def add(a, b):  return a + b'''
    embedding1 = forward_modernbert(string1, "cpu")
    embedding2 = forward_modernbert(string2, "cpu")
    print(embedding1.shape)
    print(embedding2.shape)
    # breakpoint()
    cosine_sim = cosine_similarity_pooled(embedding1[0], embedding2[0])
    print(cosine_sim.item())
    
    cosine_sim = cosine_similarity_cls(embedding1[0], embedding2[0])
    print(cosine_sim.item())
    
    cosine_sim = cosine_similarity_dense(embedding1[0], embedding2[0])
    
    print(cosine_sim.item())