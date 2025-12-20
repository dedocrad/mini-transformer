import torch

def predict(model, tokenizer, input_str, max_len=50):
    """Предсказывает перевёрнутую строку с помощью обученного трансформера"""
    model.eval()
    
    # 1. Подготовка входных данных
    src_tokens = tokenizer.encode(input_str)
    src_tensor = torch.tensor(src_tokens).unsqueeze(0)  # [1, seq_len]
    
    # 2. Создание маски для энкодера (важно!)
    src_mask = (src_tensor != tokenizer.pad_id).unsqueeze(1).unsqueeze(2)
    
    # 3. Получение представления от энкодера
    with torch.no_grad():
        encoder_out = model.encode(src_tensor, src_mask)
    
    # 4. Автогрессивная генерация
    output = [tokenizer.bos_id]  # начинаем с токена начала последовательности
    
    for i in range(max_len):
        # Подготовка текущего tgt
        tgt_tensor = torch.tensor(output).unsqueeze(0)  # [1, current_len]
        
        # Создание causal mask для декодера
        tgt_mask = (tgt_tensor != tokenizer.pad_id).unsqueeze(1).unsqueeze(2)
        tgt_mask = tgt_mask & subsequent_mask(tgt_tensor.size(1)).type_as(tgt_mask)
        
        # Предсказание следующего токена
        with torch.no_grad():
            # Важно: используем правильный метод decode с масками
            decoder_out = model.decode(tgt_tensor, encoder_out, src_mask, tgt_mask)
            # decoder_out имеет размер [1, current_len, vocab_size]
            next_token_logits = decoder_out[0, -1, :]  # логиты для последнего токена
            next_token = torch.argmax(next_token_logits, dim=-1).item()
        
        # Проверка на завершение
        if next_token == tokenizer.eos_id:
            break
            
        output.append(next_token)
    
    # 5. Декодирование результата
    return tokenizer.decode(output)


def subsequent_mask(size):
    """Создаёт маску, чтобы предотвратить просмотр будущих токенов в декодере"""
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0