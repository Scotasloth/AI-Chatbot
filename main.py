from transformers import GPT2Tokenizer, GPT2LMHeadModel

def main():
    
    token = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    input = "test"  
    input_ids = token.encode(input, return_tensors="pt")

    output = model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=token.eos_token_id)

    reply = token.decode(output[0], skip_special_tokens=True)

    print("AI", reply)

if __name__ == '__main__':
    main()