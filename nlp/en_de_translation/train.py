from torch import dropout
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import Multi30k
from typing import Iterable, List
from my_model import *
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from timeit import default_timer as timer
import os

DATPATH = '/home/wangqs/Datasets'
MODPATH = './ckpt.pt'

SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EMB_SIZE = 128
NHEAD = 2
FFN_HID_DIM = 512
BATCH_SIZE = 32
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2


# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']


# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))



# helper function to yield list of tokens
def yield_tokens(data_iter: Iterable, language: str, token_transform) -> List[str]:
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

    for data_sample in data_iter:
        yield token_transform[language](data_sample[language_index[language]])



def train_epoch(model, optimizer, loss_fn, collate_fn):
    model.train()
    losses = 0
    iter = 0
    train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for src, tgt in train_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, PAD_IDX, DEVICE)

        logits = model(src=src, tgt=tgt_input, src_mask=src_mask, tgt_mask=tgt_mask, 
            src_key_padding_mask=src_padding_mask, tgt_key_padding_mask=tgt_padding_mask, memory_key_padding_mask=src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

        iter += 1
        if iter % 200 == 0:
            print(f'| iter {iter:5d} | train loss   {loss.item():5.2f} |')
            for n in range(len(model.encoder.layers)):
                print(f'| encoder layer {n:3d} | sparsity {model.encoder.layers[n].act_spst:5.2f}')
            for n in range(len(model.decoder.layers)):
                print(f'| decoder layer {n:3d} | sparsity {model.decoder.layers[n].act_spst:5.2f}')

    return losses / iter


def evaluate(model, loss_fn, collate_fn):
    model.eval()
    losses = 0
    iter = 0

    val_iter = Multi30k(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for src, tgt in val_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, PAD_IDX, DEVICE)

        logits = model(src=src, tgt=tgt_input, src_mask=src_mask, tgt_mask=tgt_mask, 
            src_key_padding_mask=src_padding_mask, tgt_key_padding_mask=tgt_padding_mask, memory_key_padding_mask=src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

        iter += 1

    return losses / iter


# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys


# actual function to translate input sentence into target language
def translate(model: torch.nn.Module, src_sentence: str, text_transform, vocab_transform):
    model.eval()
    src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    return " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")


def main():
    # Place-holders
    token_transform = {}
    vocab_transform = {}


    # Create source and target language tokenizer. Make sure to install the dependencies.
    # pip install -U spacy
    # python -m spacy download en_core_web_sm
    # python -m spacy download de_core_news_sm
    token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='de_core_news_sm')
    token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')


    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        # Training data Iterator
        train_iter = Multi30k(root=DATPATH, split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
        # Create torchtext's Vocab object
        vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln, token_transform),
                                                        min_freq=1,
                                                        specials=special_symbols,
                                                        special_first=True)

    # Set UNK_IDX as the default index. This index is returned when the token is not found.
    # If not set, it throws RuntimeError when the queried token is not found in the Vocabulary.
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        vocab_transform[ln].set_default_index(UNK_IDX)

    torch.manual_seed(0)

    SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
    TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])

    transformer = Seq2seq_Transformer(src_vocab_size=SRC_VOCAB_SIZE, tgt_vocabsize=TGT_VOCAB_SIZE, 
        num_encoder_layers=NUM_ENCODER_LAYERS, num_decoder_layers=NUM_DECODER_LAYERS, d_model=EMB_SIZE, nhead=NHEAD, dim_feedforward=FFN_HID_DIM)

    if os.path.exists(MODPATH):
        transformer.load_state_dict(torch.load(MODPATH))

    else:
        for p in transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    transformer = transformer.to(DEVICE)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)

    # src and tgt language text transforms to convert raw strings into tensors indices
    text_transform = {}
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        text_transform[ln] = sequential_transforms(token_transform[ln], #Tokenization
                                               vocab_transform[ln], #Numericalization
                                               tensor_transform) # Add BOS/EOS and create tensor

    #function to collate data samples into batch tesors
    def collate_fn(batch):
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
            tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

        src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
        tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
        return src_batch, tgt_batch

    NUM_EPOCHS = 5

    for epoch in range(1, NUM_EPOCHS+1):
        start_time = timer()
        train_loss = train_epoch(transformer, optimizer, loss_fn, collate_fn)
        end_time = timer()
        val_loss = evaluate(transformer, loss_fn, collate_fn)
        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))

    torch.save(transformer.state_dict(), MODPATH)



if __name__ == '__main__':
    main()