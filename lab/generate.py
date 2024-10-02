
import torch
import torch.nn as nn
from tqdm import auto as tqdm_lib



class BreakOuterLoop(Exception):
    pass

def greedy_generate(m, model: nn.Module, input_ids: torch.Tensor, max_seq_len: int,
                    verbose=True):

    initial_input_length = input_ids.shape[1]
    current_input_ids = input_ids
    max_seq_len = initial_input_length + max_seq_len  # It is enough to output only 30 more tokens
    layer_past = None
    layer_past_length = 0
    all_token_ids = input_ids.tolist()
    batch_size = len(all_token_ids)
    
    trange = range(initial_input_length, max_seq_len)

    input_length = current_input_ids.shape[1]
    model_out, layer_past = model(
        current_input_ids,
        layer_past=layer_past,
    )

    top_10_indices = torch.topk(model_out[:, -1], k=10, dim=-1).indices
    greedy_predicted_token_ids = top_10_indices[:, m]  #
    current_input_ids = greedy_predicted_token_ids[:, None]
    l = []
    l.append(greedy_predicted_token_ids.item())

    try:
        should_break = False #Initialize the flag variable to False
        for _ in trange: # Specify the iteration range appropriately
            input_length = current_input_ids.shape[1]
            model_out, layer_past = model(
                current_input_ids,
                layer_past=layer_past,
            )

            greedy_predicted_token_ids = model_out[:, -1].argmax(-1)

            current_input_ids = greedy_predicted_token_ids[:, None]
            layer_past_length += input_length

            for i in range(batch_size):
                if greedy_predicted_token_ids[i].item() == 187:
                    should_break = True
                    raise BreakOuterLoop

                l.append(greedy_predicted_token_ids[i])

            if should_break:
                break

    except BreakOuterLoop:
        pass

    return l


def greedy_generate_text(m, model: nn.Module,
                         tokenizer,
                         initial_str: str,
                         max_seq_len: int,
                         device=torch.device("cuda:0"),
                         verbose=True):

    tokenized = tokenizer.encode(initial_str)
    if len(tokenized.ids) > 2020:
        input_ids = torch.LongTensor([tokenized.ids[-2020:]]).to(device)
    else:
        input_ids = torch.LongTensor([tokenized.ids]).to(device)

    try:
        all_token_ids = greedy_generate(m, model=model, input_ids=input_ids, max_seq_len=max_seq_len, verbose=verbose)
    except BreakOuterLoop:
        pass

    decoded_str = tokenizer.decode(all_token_ids)
    if len(decoded_str)< 2:
        return '"#'+str(m)+'"'
    elif decoded_str[1].isdigit():
        return tokenizer.decode(all_token_ids)
    else:
        return '"#'+str(m)+'"'











