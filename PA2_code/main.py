import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import argparse
import matplotlib.pyplot as plt

from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset

from transformer import Encoder_version1, Decoder_version1
from utilities import Utilities


seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Hyperparameters to use for training to roughly match 
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 5e-3  # Learning rate for the optimizer
#learning_rate = 2e-3
# learning_rate = 1e-3
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers


eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500 # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set


## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input 
## size of 64, hidden size of 50 and output size of 3.

n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15 # epochs for classifier training

def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data. 
    """

    texts = []
    files = os.listdir(directory)
    for filename in files: 
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts





def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)  
    return padded_sequences, labels

def compute_classifier_accuracy(classifier, data_loader):
    """ Compute the accuracy of the classifier on the data in data_loader."""
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            outputs = classifier(X)
            _, predicted = torch.max(outputs[-1].data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
        accuracy = (100 * total_correct / total_samples)
        classifier.train()
        return accuracy


def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    losses= []
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        out = decoderLMmodel(X) # your model should be computing the cross entropy loss
        loss = torch.nn.functional.cross_entropy(out[-1].view(-1, out[-1].size(-1)), Y.view(-1), reduction="mean")
        losses.append(loss.item())
        if len(losses) >= eval_iters: break


    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

    decoderLMmodel.train()
    return perplexity

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='encoder', choices=['encoder', 'decoder', 'exploration'], help='Task to perform')
    parser.add_argument('--subtask', type=str, default=None, choices=['architectural1', 'architectural2', 'performance'], help='Subtask for exploration')
    args = parser.parse_args()

    if args.task == 'encoder':
        run_encoder()
    elif args.task == 'decoder':
        run_decoder()
    elif args.task == 'exploration':
        run_exploration(args.subtask)
    def run_encoder(exploration=None):
    ############################## Encoder Task ########################################

    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)

    train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
    train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)

    test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
    test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=False)

    CLS_model = Encoder_version1(d_model=n_embd,
                                 n_heads=n_head,
                                 bias=False,
                                 dropout_attn=0,
                                 dropout_O=0,
                                 is_casual=False,
                                 n_layers=n_layer,
                                 d_hidden=n_hidden,
                                 num_classes=n_output,
                                 vocab_size=tokenizer.vocab_size,
                                 block_size=block_size,
                                 exploration=exploration).to(device)

    CLS_model.train()

    optimizer = torch.optim.AdamW(CLS_model.parameters(), lr=learning_rate, weight_decay=1e-2)
    num_params_trainabel = sum(p.numel() for p in CLS_model.parameters() if p.requires_grad)
    num_params_total = sum(p.numel() for p in CLS_model.parameters())

    print(f"Number of trainable parameters: {num_params_trainabel}")
    print(f"Total number of parameters: {num_params_total}")

    num_batches = len(train_CLS_loader)
    warmup_steps = int(0.1 * epochs_CLS * num_batches)
    total_steps = epochs_CLS * num_batches
    lr_s = []

    #  for the classification  task, you will train for a fixed number of epochs like this:
    for epoch in range(epochs_CLS):
        for idx, (xb, yb) in enumerate(train_CLS_loader):
            xb, yb = xb.to(device), yb.to(device)
            _, _, att_list, o = CLS_model(xb)
            loss = torch.nn.functional.cross_entropy(o, yb)
            loss.backward()
            step = idx + num_batches * epoch

            if step < warmup_steps:
                lr = learning_rate * step / warmup_steps
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            else:
                lr = learning_rate * ((total_steps / (total_steps - warmup_steps)) - (step / (total_steps - warmup_steps)))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            lr_s.append(lr)
            optimizer.step()
            optimizer.zero_grad()
        acc_test = compute_classifier_accuracy(CLS_model, test_CLS_loader)
        print(f"Epoch {epoch}, test accuracy: {acc_test:.2f}")
    
    
    plt.plot(lr_s)
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.savefig('./learning_rate_plot.png')
    util_obj = Utilities(tokenizer=tokenizer, model=CLS_model)
    util_obj.sanity_check("That is in Israel's interest, Palestine's interest, America's interest, and the world's interest.", block_size)
    
    ############################## End of Encoder Task ########################################

    def run_decoder(exploration=None, performance_improvement=False):
    ############################## Decoder Task ########################################
    
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts))
    print("Vocabulary size is", tokenizer.vocab_size)
    inputfile = "speechesdataset/train_LM.txt"
    with open(inputfile, 'r', encoding='utf-8') as f:
        lmtrainText = f.read()
    train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
    train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)

    politicians = {"hbush": None, "obama": None, "wbush": None}
    for politician in politicians:
        if politician != "hbush":
            with open(f"speechesdataset/test_LM_{politician}.txt", 'r', encoding='utf-8') as f:
                politicians[politician] = f.read()
        else:
            with open(f"speechesdataset/test_LM_{politician}.tsv", 'r', encoding='utf-8') as f:
                politicians[politician] = f.read()
    
    for politician in politicians:
        politicians[politician] = LanguageModelingDataset(tokenizer, politicians[politician], block_size)
        politicians[politician] = DataLoader(politicians[politician], batch_size=batch_size, shuffle=False)

    LM_model = Decoder_version1(d_model=n_embd,
                                 n_heads=n_head,
                                 bias=False,
                                 dropout_attn=0,
                                 dropout_O=0,
                                 is_casual=True,
                                 n_layers=n_layer,
                                 d_hidden=n_hidden,
                                 num_classes=n_output,
                                 vocab_size=tokenizer.vocab_size,
                                 block_size=block_size,
                                 exploration=exploration).to(device)

    if performance_improvement:
        optimizer = torch.optim.AdamW(LM_model.parameters(), lr=learning_rate, weight_decay=0.0)
    else:
        optimizer = torch.optim.AdamW(LM_model.parameters(), lr=learning_rate, weight_decay=1e-1)

    for politician in politicians:
        perplexity = compute_perplexity(LM_model, politicians[politician], eval_iters)
        print(f"Before Training, politician {politician}, perplexity: {perplexity:.2f}")

    # for the language modeling task, you will iterate over the training data for a fixed number of iterations like this:
    for i, (xb, yb) in enumerate(train_LM_loader):
        if i >= max_iters:
            break
        xb, yb = xb.to(device), yb.to(device)
        _, _, att_w, out = LM_model(xb)
        loss = torch.nn.functional.cross_entropy(out.view(-1, out.size(-1)), yb.view(-1), reduction="mean")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if i % eval_interval == 0:
            for politician in politicians:
                perplexity = compute_perplexity(LM_model, politicians[politician], eval_iters)
                print(f"Iteration {i}, politician {politician}, perplexity: {perplexity:.2f}")
            perplexity_train = compute_perplexity(LM_model, train_LM_loader, eval_iters)
            print(f"Iteration {i}, train perplexity: {perplexity_train:.2f}")
    num_params_trainabel = sum(p.numel() for p in LM_model.parameters() if p.requires_grad)
    num_params_total = sum(p.numel() for p in LM_model.parameters())

    print(f"Number of trainable parameters: {num_params_trainabel}")
    print(f"Total number of parameters: {num_params_total}")

    util_obj_2 = Utilities(tokenizer=tokenizer, model=LM_model)
    util_obj_2.sanity_check("That is in Israel's interest, Palestine's interest, America's interest, and the world's interest.", block_size)

def run_exploration(subtask):
    if subtask == 'architectural1':
        print("Running architectural exploration 1...")
        # Decoder
        run_decoder(exploration='architectural1')
        # Encoder
        run_encoder(exploration='architectural1')
    elif subtask == 'architectural2':
        print("Running architectural exploration 2...")
        # Decoder
        run_decoder(exploration='architectural2')
        # Encoder
        run_encoder(exploration='architectural2')
    elif subtask == 'performance':
        print("Running performance improvement exploration...")
        run_decoder(performance_improvement=True)
    else:
        print(f"Invalid subtask: {subtask}")

        

    



if __name__ == "__main__":
    main()