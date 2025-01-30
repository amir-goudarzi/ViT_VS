import torch
from torch import nn
from src.VisionTransformer import ViT
import utils.options as options
from utils.make_dataloader import get_loaders
import os

def main(args): 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n--- GPU Information ---\n")

    if torch.cuda.is_available():
        print(f"Model is using device: {device}")
        print(f"CUDA Device: {torch.cuda.get_device_name(device)}")
        print(f"Total Memory: {torch.cuda.get_device_properties(device).total_memory / 1024 ** 2} MB")
    else:
        print("Model is using CPU")

    params = {
    "PATCH_SIZE": args.patch_size,
    "POSITIONAL_ENCODING": args.positional_encoding,
    "EMBEDDING_DIM": args.embedding_dim,
    "NUM_TRANSFORMER_LAYERS": args.num_transformer_layers,
    "MLP_DROPOUT": args.mlp_dropout,
    "ATTN_DROPOUT": args.attn_dropout,
    "MLP_SIZE": args.mlp_size,
    "NUM_HEADS": args.num_heads,
    "BATCH_SIZE": args.batch_size,
    "OPTIMIZER": args.optimizer,
    "LEARNING_RATE": args.learning_rate,
    "NUM_EPOCHS": args.num_epochs,
    "NUM_WORKERS": args.num_workers
}

    train_loader, val_loader, test_loader, n_classes = get_loaders(batch_size= params['BATCH_SIZE'], 
                                                                   num_workers=params['NUM_WORKERS'], 
                                                                   path= os.path.join(args.dir, args.dataset_path), 
                                                                   split=args.split,
                                                                   return_whole_puzzle=False)    
    model = ViT(freq_encoding=args.freq_encoding,
                embedding_dim=args.embedding_dim,
                num_transformer_layers=args.num_transformer_layers,
                mlp_dropout=args.mlp_dropout,
                attn_dropout=args.attn_dropout,
                mlp_size=args.mlp_size,
                num_heads=args.num_heads,
                num_classes=n_classes).to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    if args.adam_optimizer:
        optimizer = torch.optim.Adam(model.parameters(), lr= args.learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr= args.learning_rate)

    def train(epoch):

        correct = 0
        total = 0

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            x, _, sudoku_label = batch
            x, sudoku_label = x.to(device), sudoku_label.to(device)
            y = model(x)
            loss = loss_fn(y, sudoku_label)
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch+1}, Batch {batch_idx + 1}, Loss: {loss.item():.2f}')


        for batch_idx, batch in enumerate(val_loader):
            x, _, sudoku_label = batch
            x, sudoku_label = x.to(device), sudoku_label.to(device)
            y = model(x)
            predictions = torch.argmax(y, 1)
            total += x.shape[0]
            correct += predictions.eq(sudoku_label).sum().item()

        acc = (correct / total) * 100

        print(f'Epoch {epoch+1}, acc: {acc:.2f}')


    def test():
        model.eval()
        correct = 0
        total = 0
        for batch in test_loader:
            x, _, sudoku_label = batch
            x, sudoku_label = x.to(device), sudoku_label.to(device)
            y = model(x)
            predictions = torch.argmax(y, 1)
            total += x.shape[0]
            correct += predictions.eq(sudoku_label).sum().item()

        acc = (correct / total) * 100

        print(f'\n\nTest accuracy: {acc:.2f}')

    for epoch in range(args.num_epochs):
        train(epoch)

    test()

if __name__ == "__main__":
    args = options.read_command_line()
    main(args)