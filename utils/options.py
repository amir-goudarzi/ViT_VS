import argparse

def read_command_line():

    parser = argparse.ArgumentParser(description='Vision Transformer')


    # Model
    parser.add_argument('--patch_size', type=int, required=False, default=28, help="Size of each token")
    parser.add_argument('--positional_encoding', type=str, required=False, default=True, choices=['learned', 'absolute', 'frequency'], 
                        help='positional encoding')
    parser.add_argument('--embedding_dim', type=int, required=False, default= 256, help='Embedding dimenstion for the entire model')
    parser.add_argument('--num_transformer_layers', type=int, required=False, default= 3, help='The number of transformer encoder layers')
    parser.add_argument('--mlp_dropout', type=float, required=False, default=0.1, help='Probability of dropout in MLP')
    parser.add_argument('--attn_dropout', type=float, required=False, default=0.0, help='Probability of dropout in attention')
    parser.add_argument('--mlp_size', type=int, required=False, default=256, help='Hidden dimension in MLP')
    parser.add_argument('--num_heads', type=int, required=False, default=4, help='Number of attention heads')
    
    # Training 
    parser.add_argument('--batch_size', type=int, required=False, default=10, help='Batch size')
    parser.add_argument('--optimizer', type=str, required=False, default='Adam', choices=['Adam', 'SGD'], help='Adam or SGD')
    parser.add_argument('--learning_rate', type=float, required=False, default=0.0001, help='Learning rate')
    parser.add_argument('--scheduler', type= bool, required=False, default=False, help='True for using a learning rate scheduler')
    parser.add_argument('--num_epochs', type=int, required=False, default=10, help='Number of epochs')
    parser.add_argument('--num_workers', type=int, required=False, default=1, help='Number of workers')


    # Others
    parser.add_argument('--dir', type=str, default='.',
                    help='Data directory')
    parser.add_argument('--dataset_path', default='VisualSudoku', type=str, help='The path to the training data.')
    parser.add_argument("--neptune_project", type=str, help="Neptune project directory")
    parser.add_argument("--neptune_api_token", type=str, help="Neptune api token")
    parser.add_argument("--split", type=int, required=False, default=10, help="Data split")


    args = parser.parse_args()
    return args