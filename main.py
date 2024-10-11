import torch
from torch import nn
from VisionTransformer import ViT
import options
from make_dataloader import get_loaders

def main(configs): 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, test_loader, n_classes = get_loaders(batch_size= configs.batch_size)
    
    model = ViT(freq_encoding=configs.freq_encoding,
                embedding_dim=configs.embedding_dim,
                num_transformer_layers=configs.num_transformer_layers,
                mlp_dropout=configs.mlp_dropout,
                attn_dropout=configs.attn_dropout,
                mlp_size=configs.mlp_size,
                num_heads=configs.num_heads,
                num_classes=n_classes).to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    if configs.adam_optimizer:
        optimizer = torch.optim.Adam(model.parameters(), lr= configs.learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr= configs.learning_rate)

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

        print(f'Test accuracy: {acc:.2f}')

    for epoch in range(configs.num_epochs):
        train(epoch)

    test()

if __name__ == "__main__":
    configs = options.read_command_line()
    main(configs)