import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn
from transformers import TextDataset, GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, DataCollatorForLanguageModeling
import json
from tokenizers import ByteLevelBPETokenizer
class LineByLineTextDataset(Dataset):
    def __init__(self, file_path, tokenizer: GPT2Tokenizer):
        self.file_path = file_path
        self.tokenizer = tokenizer
        with open(self.file_path, 'r', encoding='utf-8') as file:
            file.seek(0)
            lines = [tokenizer.encode(i.strip(), return_tensors="pt") for i in file.readlines() if len(i) > 3]

        self.lines = lines
        print(lines)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]

        return line
def train_model(base_model, block_size=512, fp_of_train_data="text.txt", fp_to_save="pretrained", lr=0.00002, epochs=5, optimc="AdamW", dataset_type = "default"):
    if dataset_type not in ("default", "linebyline"):
        raise ValueError("the `dataset_type` argument must be `default` or `linebyline`")
    tokenizer = GPT2Tokenizer.from_pretrained(base_model)
    model = GPT2LMHeadModel.from_pretrained(base_model)
    if dataset_type == "default":
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
        model.train()
        for i in range(epochs):
            for b, text in enumerate(dataloader):
                optim.zero_grad()
                input_text = torch.clone(text)[0][:-1][None, :]

                out = model(input_text)
                target = text[0][1:].long()
                loss = criterion(out.logits[0], target)
                loss.backward()
                optim.step()
                print(f"Batch {b + 1}/{dataset[:].size()[0]}. Loss: {loss.item()}")
            print("-" * 89)
            print(f"Epoch {i+1}/{epochs}. Loss: {loss.item()}")
            print("-" * 89)
            model.save_pretrained(fp_to_save)
            tokenizer.save_pretrained(fp_to_save)
        print("DONE")
    else:
        dataset = LineByLineTextDataset(fp_of_train_data, tokenizer)
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
        model.train()
        for i in range(epochs):
            for b, text in enumerate(dataset):
                optim.zero_grad()
                input_text = torch.clone(text)[0][:-1][None, :]
                out = model(input_text)
                target = text[0][1:].long()
                loss = criterion(out.logits[0], target)
                loss.backward()
                optim.step()
                print(f"Batch {b + 1}/{len(dataset)}. Loss: {loss.item()}")
            print("-" * 89)
            print(f"Epoch {i + 1}/{epochs}. Loss: {loss.item()}")
            print("-" * 89)
            model.save_pretrained(fp_to_save)
            tokenizer.save_pretrained(fp_to_save)
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

def create_custom_model(name, tokenizer, n_ctx, n_embd, n_layer, n_head):
    custom_config = GPT2Config(
        vocab_size=len(tokenizer),  # размер словаря
        n_positions=n_ctx,  # максимальное количество позиций
        n_ctx=n_ctx,  # контекст
        n_embd=n_embd,  # размер эмбеддинга
        n_layer=n_layer,  # количество слоев
        n_head=n_head  # количество голов в механизме внимания
    )
    model = GPT2LMHeadModel(config=custom_config)
    model.resize_token_embeddings(len(tokenizer))
    print(f"Total model parameters: {sum(p.numel() for p in model.parameters())}")
    model.save_pretrained(name)
    tokenizer.save_pretrained(name)


def create_custom_model_and_train(name, tokenizer, n_ctx, n_embd, n_layer, n_head, block_size=512, fp_of_train_data="text.txt", lr=0.00002, epochs=5, optimc="AdamW"):
    create_custom_model(name, tokenizer, n_ctx, n_embd, n_layer, n_head)
    train_model(name, block_size, fp_of_train_data, name, lr, epochs, optimc)

def create_custom_tokenizer(name, ctx, max_vocab_size, min_frequency, fp_train_file):
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=fp_train_file, vocab_size=max_vocab_size, min_frequency=min_frequency)

    special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    tokenizer.add_special_tokens(special_tokens)

    tokenizer.save_model(name)
    # Параметры вашего токенизатора
    tokenizer_config = {
        "model_max_length": ctx,
        "vocab_size": tokenizer.get_vocab_size(),  # Замените на количество слов в вашем словаре
        "do_lower_case": False,  # Измените в соответствии с вашими настройками
        "model_input_names": ["input_ids", "attention_mask"]
    }

    path_to_save_config = f"{name}/tokenizer_config.json"
    with open(path_to_save_config, "w") as file:
        json.dump(tokenizer_config, file, indent=4)
    print("Tokenizer created")
