from transformers import GPT2Tokenizer, GPT2LMHeadModel
import sys
import torch

def main():
    dir = sys.path[0]

    model_path = "Childebot.pth"
    model = torch.load(model_path)
    
    token = GPT2Tokenizer.from_pretrained("Childebot.py")

    input = "Hey"  
    input_ids = token.encode(input, return_tensors="pt")

    output = model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=token.eos_token_id)

    reply = token.decode(output[0], skip_special_tokens=True)

    print("AI", reply)

if __name__ == '__main__':
    main()