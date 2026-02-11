import torch

def predict(model, tokenizer, input_str, max_len=50):
    """Предсказывает перевёрнутую строку с помощью обученного трансформера"""
    model.eval()
    
    src_tokens = tokenizer.encode(input_str)
    src_tensor = torch.tensor(src_tokens).unsqueeze(0)  # [1, seq_len]
    
    src_mask = (src_tensor != tokenizer.pad_id).unsqueeze(1).unsqueeze(2)
    
    with torch.no_grad():
        encoder_out = model.encode(src_tensor, src_mask)

    output = [tokenizer.bos_id]
    
    for i in range(max_len):
        tgt_tensor = torch.tensor(output).unsqueeze(0)  # [1, current_len]
        
        # Создание causal mask для декодера
        tgt_mask = (tgt_tensor != tokenizer.pad_id).unsqueeze(1).unsqueeze(2)
        tgt_mask = tgt_mask & subsequent_mask(tgt_tensor.size(1)).type_as(tgt_mask)
        
        # предсказание следующего токена
        with torch.no_grad():
            decoder_out = model.decode(tgt_tensor, encoder_out, src_mask, tgt_mask)
            next_token_logits = decoder_out[0, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).item()
        
        if next_token == tokenizer.eos_id:
            break
            
        output.append(next_token)
    
    return tokenizer.decode(output)


def subsequent_mask(size):
    """Создаёт маску, чтобы предотвратить просмотр будущих токенов в декодере"""
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0