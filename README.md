# VisualSudoku

## Installation

1. Clone this repository to your machine:
   ```bash
   git clone https://github.com/Amir-Goudarzi/VisualSudoku
2. Navigate to the project directory:
   ```bash
   cd VisualSudoku
3. Install the required Python packages from requirements.txt:
   ```bash
   pip install -r requirements.txt

## Usage and Configuration 

   Use the following script to specify the parameters and train the model. 

   ```bash
   FREQ_ENCODING=True
   EMBEDDING_DIM=256
   NUM_TRANSFORMER_LAYERS=3
   MLP_DROPOUT=0.1
   ATTN_DROPOUT=0.0
   MLP_SIZE=256
   NUM_HEADS=4
   BATCH_SIZE=10
   ADAM_OPTIMIZER=True
   LEARNING_RATE=0.0001
   NUM_EPOCHS=10
   NUM_WORKERS=1
   
   
   python ./main.py \
      --freq_encoding "$FREQ_ENCODING" \
      --embedding_dim "$EMBEDDING_DIM" \
      --num_transformer_layers "$NUM_TRANSFORMER_LAYERS" \
      --mlp_dropout "$MLP_DROPOUT" \
      --attn_dropout "$ATTN_DROPOUT" \
      --mlp_size "$MLP_SIZE" \
      --num_heads "$NUM_HEADS" \
      --batch_size "$BATCH_SIZE" \
      --adam_optimizer "$ADAM_OPTIMIZER" \
      --learning_rate "$LEARNING_RATE" \
      --num_epochs "$NUM_EPOCHS" \
      --num_workers "$NUM_WORKERS"
