import itertools
import subprocess


# Hyperparameter ranges



# Fixed parameters
patch_size = 16


# Loop over each combination
for params in param_combinations:

    patch_window_combination, mlp_size, num_transformer_layer, num_attention_head, embedding_dim, learning_rate, optimizer, batch_size = params

    vit_mlp_ratio = int(mlp_size / embedding_dim)
    patch_size, window_size = patch_window_combination

    # Construct command
    command = f"""
    python ./main.py \
      --patch_size "{patch_size}" \
      --positional_encoding "{positional_encoding}" \
      --embedding_dim "{embedding_dim}" \
      --num_transformer_layers "{num_transformer_layers}" \
      --mlp_dropout "{mlp_dropout}" \
      --attn_dropout "{attn_dropout}" \
      --mlp_size "{mlp_size}" \
      --num_heads "{num_heads}" \
      --batch_size "{batch_size}" \
      --optimizer "{optimizer}" \
      --learning_rate "{learning_rate}" \
      --num_epochs "{num_epochs}" \
      --num_workers "{num_workers}"
    """
    
    