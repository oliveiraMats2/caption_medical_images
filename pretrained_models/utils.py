from tqdm import tqdm

def translate_encoded_ids(encoded_ids_list, tokenizer):
    phrases = []
    for encoded_ids in encoded_ids_list:
        decoded_ids = tokenizer.decode(encoded_ids, skip_special_tokens=True)
        phrases.append(decoded_ids)
    return phrases

def generate_evaluation_dict(model, dataloader, tokenizer, n_eval=100, dataset=False, run_entire_dataset=False, device="cpu"):
    model.eval()
    batch_size = dataloader.batch_size

    if run_entire_dataset is False:

        if n_eval%batch_size:
            raise Exception(f"[ERROR] n_eval must be divisble by batch_size. Current n_eval: {n_eval}, batch_size: {batch_size}")

        n_iter = int(n_eval/batch_size)
        iter_loader = iter(dataloader)

        results_dict = {}
        for i in tqdm(range(n_iter)):
            img, _, _, _, img_name = next(iter_loader)
            img = img.to(device)
            prediction = model(img)
            decoded_prediction = translate_encoded_ids(prediction, tokenizer)
            name_prediction = zip(img_name, decoded_prediction)
            for name, prediction in name_prediction:
                results_dict[name] = prediction
    
    else:
        results_dict = {}
        for img, _, _, _, img_name in dataset:
            img = img.to(device)
            prediction = model(img.unsqueeze(0))
            decoded_prediction = translate_encoded_ids(prediction, tokenizer)
            name_prediction = zip(img_name, decoded_prediction)
            for name, prediction in name_prediction:
                results_dict[name] = prediction
    
    return results_dict
