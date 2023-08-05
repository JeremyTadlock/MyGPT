import BigramGPT
import torch
import gpt_tokenizers

# Create bigram gpt model
model_path = 'BPE_7500_2_12m_8000s.pth'
model = BigramGPT.BigramLanguageModel()
m = model.to(BigramGPT.device)  # CUDA!!1!1
m.load_state_dict(torch.load(model_path))
model.eval()

 # Load model tokenizer
 #rokenizer = gpt_tokenizers.byte





def load_batch(split):w
 data = train_data if split == 'train' else val_data
 ix = torch.randint(len(data) - block_size, (batch_size,))
 x = torch.stack([data[i:i + block_size] for i in ix])
 y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
 x, y = x.to(device), y.to(device)
 return x, y
@torch.no_grad()  # tell's pytorch we AREN'T using backpropagation, saving memory
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = load_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out















