Train GPT tools, this collection of functions was originally created for personal use, but still I decided to post it online. There are 5 functions in the collection:
1. train_model trains and saves the model
Parameters: base_model, block_size=512, fp_of_train_data="text.txt ", fp_to_save="pretrained", lr=0.00002, epochs=5, optimc="AdamW"

2. model_generate is a simple function created to generate the text of the selected model
Parameters: model, text, max_new_tokens, no_repeat_ngram_size, do_sample: book, **kwargs (temperature (if do_sample=True) and top_k else)

3. create_custom_model function for creating a custom model and saving it to a file
Parameters: name, tokenize (GPT2Tokenizer), n_ctx, n_embed, n_layer, n_head

4. create_custom_model_and_train creates a custom model and immediately trains it
Parameters: name, tokenize, n_cts, n_embed, n_layer, n_head, block_size=512, fp_of_train_data="text.txt ", fp_to_save="pretrained", lr=0.00002, epochs=5, optimc="AdamW"

5. And the last fifth function create_custom_tokenizer creates a custom BPE tokenizer
Parameters: name, ctx, max_vocab_size, min_frequency, fp_train_file
