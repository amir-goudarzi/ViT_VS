import argparse

def read_command_line():

    parser = argparse.ArgumentParser(description='Vision Transformer')

    parser.add_argument('--embedding_dim', type=int, required=False, default= 256, help='Embedding dimenstion for the entire model')
    parser.add_argument('--num_transformer_layers', type=int, required=False, default= 3, help='The number of transformer encoder layers')
    parser.add_argument('--mlp_dropout', type=float, required=False, default=0.1, help='Probability of dropout in MLP')
    parser.add_argument('--attn_dropout', type=float, required=False, default=0.0, help='Probability of dropout in attention')
    parser.add_argument('--mlp_size', type=int, required=False, default=256, help='Hidden dimension in MLP')
    parser.add_argument('--num_heads', type= float, required=False, default=4, help='Number of attention heads')
    parser.add_argument('--batch_size', type= int, required=False, default=10, help='Batch size')
    parser.add_argument('--adam_optimizer', type= bool, required=False, default=True, help='True for Adam, False for SGD')
    parser.add_argument('--learning_rate', type= float, required=False, default=0.0001, help='Learning rate')
    parser.add_argument('--num_epochs', type= int, required=False, default=10, help='Number of epochs')


    args = parser.parse_args()
    return args