

# export PYTHONPATH=$(pwd)
PYTHONFILE='../cangjie/training.py'

NET="resnet50" 
GPU=1
BATCH_SIZE=4
WARMUP=1
LEARNING_RATE=1e-4

python $PYTHONFILE -net $NET -gpu $GPU -batch $BATCH_SIZE -warmup $WARMUP -lr $LEARNING_RATE
