import torch
from torch import nn
from src.VisionTransformer import ViT
import utils.options as options
from utils.scheduler import build_scheduler
from utils.make_dataloader import get_loaders
import neptune.new as neptune
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
    
    model = ViT(encoding=args.positional_encoding,
                embedding_dim=args.embedding_dim,
                num_transformer_layers=args.num_transformer_layers,
                mlp_dropout=args.mlp_dropout,
                attn_dropout=args.attn_dropout,
                mlp_size=args.mlp_size,
                num_heads=args.num_heads,
                num_classes=n_classes,
                batch_size=args.batch_size).to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr= args.learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr= args.learning_rate, momentum=0.9)

    if args.scheduler:
        scheduler = build_scheduler(optimizer, lr=params["LEARNING_RATE"])

    run = neptune.init_run(
            project=args.neptune_project,
            api_token=args.neptune_api_token,
            
        )
    run["parameters"] = params

    def train(epoch):

        correct = 0
        total = 0
        correct_train = 0
        total_train = 0

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            x, _, sudoku_label = batch
            x, sudoku_label = x.to(device), sudoku_label.to(device)
            y = model(x)
            loss = loss_fn(y, sudoku_label)
            loss.backward()
            optimizer.step()
            predictions = torch.argmax(y, 1)
            total_train += x.shape[0]
            correct_train += predictions.eq(sudoku_label).sum().item()

            run[f"train/loss"].log(loss.item())

        train_accuracy = ((correct_train / total_train) * 100)

        print(f'\n\n-----Epoch {epoch+1}, Train accuracy: {train_accuracy:.2f}-----')
        run[f"train/accuracy"].log(train_accuracy)


        for batch_idx, batch in enumerate(val_loader):
            x, _, sudoku_label = batch
            x, sudoku_label = x.to(device), sudoku_label.to(device)
            y = model(x)
            loss = loss_fn(y, sudoku_label)
            predictions = torch.argmax(y, 1)
            total += x.shape[0]
            correct += predictions.eq(sudoku_label).sum().item()

            run[f"val/loss"].log(loss.item())

        acc = (correct / total) * 100

        run[f"val/accuracy"].log(acc)

        print(f'-----Val accuracy: {acc:.2f}-----\n\n')

        if args.scheduler:
            scheduler.step()


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

        print(f'\n\n\n\n--------Test accuracy: {acc:.2f}-----\n\n\n\n')
        run[f"test/accuracy"].log(acc)

    print("\n\n--Started Training--\n\n")

    for epoch in range(args.num_epochs):
        train(epoch)

    test()
    run.stop()

if __name__ == "__main__":
    args = options.read_command_line()
    main(args)