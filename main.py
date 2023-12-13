import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn
from transformers import TextDataset, GPT2Tokenizer, GPT2LMHeadModel


def train_model(base_model, block_size=512, fp_of_train_data="text.txt", fp_to_save="pretrained", lr=0.00002, epochs=5, optimc="AdamW"):
    tokenizer = GPT2Tokenizer.from_pretrained(base_model)
    model = GPT2LMHeadModel.from_pretrained(base_model)
    dataset = TextDataset(tokenizer=tokenizer, file_path=fp_of_train_data, block_size=block_size)
    dataloader = DataLoader(dataset)
    optims = {
        "AdamW": torch.optim.AdamW,
        "Adam": torch.optim.Adam,
        "Adagrad": torch.optim.Adagrad,
        "RMSprop": torch.optim.RMSprop,
        "SparseAdam": torch.optim.SparseAdam,
        "SGD": torch.optim.SGD,
        "Adadelta": torch.optim.Adagrad,
        "RAdam": torch.optim.RAdam,
        "NAdam": torch.optim.NAdam
    }
    optim = optims[optimc](model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    for i in range(epochs):
        for i, text in enumerate(dataloader):
            optim.zero_grad()
            input_text = torch.clone(text)[0][:-1][None, :]
            out = model(input_text)
            target = text[0][1:].long()
            loss = criterion(out.logits[0], target)
            loss.backward()
            optim.step()
            print(f"Batch {i + 1}/{dataset[:].size()[0]}. Loss: {loss.item()}")
            model.save_pretrained(fp_to_save)
        print("-" * 89)
        print(f"Epoch {i}/{epochs}. Loss: {loss.item()}")
        print("-" * 89)
        model.save_pretrained(fp_to_save)
        print("DONE")


def model_generate(model, text, max_new_tokens, no_repeat_ngram_size, do_sample: bool, **kwargs):
    tokenizer = GPT2Tokenizer.from_pretrained(model)
    model = GPT2LMHeadModel.from_pretrained(model)
    if do_sample == True:
        input_ids = tokenizer.encode(text, return_tensors="pt")
        output = model.generate(input_ids, max_new_tokens=max_new_tokens, no_repeat_ngram_size=no_repeat_ngram_size, do_sample=do_sample, temperature=kwargs["temperature"])[0]
        output = tokenizer.decode(output, skip_special_tokens=True)
        return output
    else:
        input_ids = tokenizer.encode(text, return_tensors="pt")
        output = model.generate(input_ids, max_new_tokens=max_new_tokens, no_repeat_ngram_size=no_repeat_ngram_size,
                                do_sample=do_sample, top_k=kwargs["top_k"])[0]
        output = tokenizer.decode(output, skip_special_tokens=True)
        return output


