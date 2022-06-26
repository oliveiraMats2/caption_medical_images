import torch
import yaml
from transformers import AutoTokenizer
from einops import repeat
import numpy as np
from constants import *

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_cased')

with open(f'{ABS_PATH}/config_models/config_decoder.yaml', 'r') as file:
    parameters = yaml.safe_load(file)

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

device = torch.device(dev)
print('Using {}'.format(device))


class Embedder(torch.nn.Module):
    def __init__(self, max_seq_length: int = parameters['max_seq_length'],
                 vocab_size: int = tokenizer.vocab_size,
                 embbeding=parameters['embbeding'],
                 pad_token_id: int = tokenizer.pad_token_id):
        super().__init__()

        self.embed = torch.nn.Embedding(vocab_size,
                                        embbeding,
                                        padding_idx=pad_token_id)

        self.embed_pos = torch.nn.Embedding(max_seq_length,
                                            embbeding,
                                            padding_idx=pad_token_id)

    def forward(self, x):
        return self.embed(x) + self.embed_pos.weight


class MultiHeadSelfAttention(torch.nn.Module):

    def __init__(self, embedding_dim=parameters['embbeding'],
                 forward_drop_p=parameters['forward_drop_p'],
                 max_seq_length: int = parameters['max_seq_length'],
                 num_heads: int = parameters['num_heads'],
                 padding_id: int = tokenizer.pad_token_id):
        super().__init__()

        self.num_heads = num_heads
        self.padding_id = padding_id
        self.max_seq_length = max_seq_length

        self.W_q = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W_k = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W_v = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.W_o = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)

    @staticmethod
    def self_attention(q, k, v, mask=None):
        """Tenho a ponderação de todos os tokens contra todos"""
        s = torch.matmul(q, k.transpose(-2, -1))
        """preenchimento da mascara"""
        if mask is not None:
            s = s.masked_fill(mask.unsqueeze(1) == 0, -float("inf"))

        """mascara aplicando softmax"""
        p = torch.nn.functional.softmax(s, dim=-1)

        att_output = torch.matmul(p, v)

        return att_output.transpose(1, 2)

    def forward(self, inputs, embbeding_V, embbeding_K, mask=None):
        batch_size = inputs.shape[0]

        """
        Modificar o codigo aqui para entrar o embbeding do modelo encoder.
        """
        inputs = repeat(inputs[:, -1, :], 'b c -> b r c', r=self.max_seq_length)

        q = self.W_q(inputs)

        q = q.view(batch_size, self.max_seq_length, self.num_heads, -1)
        q = q.transpose(1, 2)

        k = self.W_k(embbeding_K)
        k = k.view(batch_size, self.max_seq_length, self.num_heads, -1)
        k = k.transpose(1, 2)

        v = self.W_v(embbeding_V)
        v = v.view(batch_size, self.max_seq_length, self.num_heads, -1)
        v = v.transpose(1, 2)

        """
        The embbedings  K V representions the images,
        should be same dims that Q.
        """

        att_output = self.self_attention(q, k, v, mask)

        att_output = att_output.reshape(batch_size, self.max_seq_length, -1)
        att_output = self.W_o(att_output)

        return att_output


class TransformerDecoderBlock(torch.nn.Module):
    def __init__(self,
                 depth: int = 2,
                 forward_expansion=parameters['forward_expansion'],
                 forward_expansion_p=parameters['forward_drop_p'],
                 emb_size: int = parameters['embbeding'],
                 drop_p: float = parameters['drop_p'],
                 **kwargs):
        super().__init__()

        self.layer_norm = torch.nn.LayerNorm(emb_size)
        self.multi_head_self_attention = MultiHeadSelfAttention(emb_size, **kwargs)
        self.dropout = torch.nn.Dropout(drop_p)

    def forward(self, inputs, embbeding_V, embbeding_K, mask=None):
        residue = inputs

        x = self.layer_norm(inputs)
        x = self.multi_head_self_attention(x, embbeding_V, embbeding_K, mask)
        x = self.dropout(x)

        x = x + residue

        return x


class TransformerDecoder(torch.nn.Module):
    def __init__(self,
                 **kwargs):
        super().__init__()

        self.layer = TransformerDecoderBlock(**kwargs)

        self.layer_2 = TransformerDecoderBlock(**kwargs)

        self.layer_3 = TransformerDecoderBlock(**kwargs)

    def forward(self, inputs, embbeding_V, embbeding_K, mask=None):
        x = self.layer(inputs, embbeding_V, embbeding_K, mask)

        x = self.layer_2(x, embbeding_V, embbeding_K, mask)

        x = self.layer_3(x, embbeding_V, embbeding_K, mask)

        return x


class CreateMask(torch.nn.Module):
    def __init__(self,
                 max_seq_length: int = parameters['max_seq_length'],
                 padding_id: int = tokenizer.pad_token_id):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.padding_id = padding_id

    def forward(self, inputs):
        batch_size = inputs.shape[0]

        mask = torch.tril(torch.ones(batch_size, self.max_seq_length, self.max_seq_length)).to(device)
        mask = mask.masked_fill(inputs.unsqueeze(1) == self.padding_id, 0)

        return mask


class Classifier(torch.nn.Sequential):
    def __init__(self,
                 emb_size: int = parameters['embbeding'],
                 vocab_size: int = tokenizer.vocab_size):
        super().__init__(torch.nn.Linear(emb_size, emb_size * 2),
                         torch.nn.ReLU(),
                         torch.nn.Linear(emb_size * 2, vocab_size, bias=False))


class LanguageModel(torch.nn.Sequential):

    def __init__(self, vocab_size: int,
                 max_seq_length: int = parameters['max_seq_length'],
                 dim: int = parameters['embbeding'],
                 pad_token_id: int = tokenizer.pad_token_id):
        super().__init__()

        self.pad_token_id = pad_token_id

        self.mask = CreateMask(max_seq_length, pad_token_id)

        self.embedder = Embedder(max_seq_length, vocab_size, dim, pad_token_id)

        self.transformer_decoder = TransformerDecoder(depth=parameters['depth'],
                                                      emb_size=parameters['embbeding'],
                                                      drop_p=parameters['drop_p'],
                                                      forward_expansion=parameters['forward_expansion'],
                                                      forward_drop_p=parameters['forward_drop_p'],
                                                      max_seq_length=parameters['max_seq_length'],
                                                      num_heads=parameters['num_heads'],
                                                      padding_id=tokenizer.pad_token_id)

        self.classifier = Classifier(dim, vocab_size)

    def forward(self, inputs, embbeding_V, embbeding_K):
        mask = self.mask.forward(inputs)

        inputs_embbedings = self.embedder(inputs)

        h = inputs_embbedings

        output_decoder = self.transformer_decoder(inputs_embbedings, inputs_embbedings, inputs_embbedings, mask)

        inputs_embbedings = output_decoder + h

        h = inputs_embbedings

        output_transformers = self.transformer_decoder(inputs_embbedings, embbeding_V, embbeding_K)

        output_transformers = output_transformers + h

        outputs = self.classifier(output_transformers)

        return outputs


if __name__ == "__main__":

    from typing import List
    from tqdm.notebook import tqdm
    from torch.utils.data import DataLoader


    def tokenize(text: str, tokenizer):
        # Recomenda-se usar o tokenizer.batch_encode_plus pois é mais rápido.
        return tokenizer(text, return_tensors=None, add_special_tokens=False).input_ids


    class MyDataset:
        def __init__(self, texts: List[str], tokenizer, max_seq_length: int):
            self.max_seq_length = max_seq_length
            self.tokenized_texts = []
            for text in tqdm(texts):
                tokenized_text = tokenize(f'[CLS] {text}', tokenizer)
                tokenized_text += [tokenizer.vocab['[PAD]']] * max(0, 1 + max_seq_length - len(tokenized_text))

                for i in range(0, len(tokenized_text) - 1, max_seq_length):

                    if i + max_seq_length < len(tokenized_text):
                        self.tokenized_texts.append(tokenized_text[i: i + max_seq_length + 1])
                    else:
                        self.tokenized_texts.append(tokenized_text[-max_seq_length - 1:])

            self.tokenized_texts = torch.LongTensor(self.tokenized_texts)

        def __len__(self):
            return len(self.tokenized_texts)

        def __getitem__(self, idx):
            x_y = self.tokenized_texts[idx]
            return x_y[:-1], x_y[1:]


    seq_length = parameters['max_seq_length']

    max_seq_length = seq_length

    texts = open('transformer_decoder/sample-1gb.txt').readlines()

    len_max = int(len(texts) / 200)

    train_examples = int(len_max * 0.6)
    valid_examples = int(len_max * 0.3)
    test_examples = int(len_max * 0.1)

    print(f"train examples: {train_examples}")
    print(f"valid examples: {valid_examples}")
    print(f"test examples: {test_examples}")

    print(f'Read {len(texts)} lines.')

    max_lines = train_examples + valid_examples + test_examples
    print(f'Truncating to {max_lines} lines.')
    texts = texts[:max_lines]

    training_texts = texts[:-(valid_examples + test_examples)]
    valid_texts = texts[-(valid_examples + test_examples):-test_examples]
    test_texts = texts[-test_examples:]

    training_dataset = MyDataset(texts=training_texts, tokenizer=tokenizer, max_seq_length=max_seq_length)
    valid_dataset = MyDataset(texts=valid_texts, tokenizer=tokenizer, max_seq_length=max_seq_length)
    test_dataset = MyDataset(texts=test_texts, tokenizer=tokenizer, max_seq_length=max_seq_length)

    model = LanguageModel(
        vocab_size=tokenizer.vocab_size,
        max_seq_length=max_seq_length,
        dim=64,
        pad_token_id=tokenizer.pad_token_id,
    ).to(device)

    sample_input, _ = next(iter(DataLoader(training_dataset, batch_size=1)))
    sample_input = sample_input.to(device)
    sample_output = model(sample_input)
    print(f'sample_input.shape: {sample_input.shape}')
    print(f'sample_output.shape: {sample_output.shape}')

    max_examples = 150_000_000
    eval_every_steps = 1000
    lr = 3e-4

    train_loader = DataLoader(training_dataset, batch_size=255, shuffle=True, drop_last=True)
    validation_loader = DataLoader(valid_dataset, batch_size=255)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           factor=0.2,
                                                           patience=0,
                                                           verbose=True)


    def train_step(input_ids, target_ids):

        model.train()
        model.zero_grad()

        logits = model(input_ids)
        logits = logits.reshape(-1, logits.shape[-1])

        target_ids = target_ids.reshape(-1)

        loss = torch.nn.functional.cross_entropy(logits, target_ids, ignore_index=model.pad_token_id)
        loss.backward()

        optimizer.step()

        return loss.item()


    def validation_step(input_ids, target_ids):

        model.eval()

        logits = model(input_ids)
        logits = logits.reshape(-1, logits.shape[-1])

        target_ids = target_ids.reshape(-1)

        loss = torch.nn.functional.cross_entropy(logits, target_ids, ignore_index=model.pad_token_id)

        return loss.item()


    train_losses = []
    n_examples = 0
    step = 0

    print("Training inicialize")

    while n_examples < max_examples:

        for train_input_ids, train_target_ids in tqdm(train_loader):

            loss = train_step(train_input_ids.to(device), train_target_ids.to(device))
            train_losses.append(loss)

            if step % eval_every_steps == 0:
                train_ppl = np.exp(np.average(train_losses))

                with torch.no_grad():
                    valid_ppl = np.exp(np.average([
                        validation_step(val_input_ids.to(device), val_target_ids.to(device))
                        for val_input_ids, val_target_ids in validation_loader]))

                print(
                    f'{step} steps; {n_examples} examples so far; train ppl: {train_ppl:.2f}, valid ppl: {valid_ppl:.2f}'
                )

                train_losses = []

                scheduler.step(valid_ppl)

            n_examples += len(train_input_ids)  # Increment of batch size
            step += 1
            if n_examples >= max_examples:
                break
