

# export PYTHONPATH=$(pwd)
PYTHONFILE='../cangjie/training.py'

NET="resnet-m" 
GPU=1
BATCH_SIZE=128
WARMUP=1
LEARNING_RATE=1e-4
CHECKPOINT="checkpoints/resnet-m"

python $PYTHONFILE -net $NET -gpu $GPU -batch $BATCH_SIZE -warmup $WARMUP -lr $LEARNING_RATE -checkpoint_dir $CHECKPOINT
