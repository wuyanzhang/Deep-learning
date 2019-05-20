import sys
from train import *
from evaluate import *
if __name__ == '__main__' and sys.argv[1] == 'train':
    run_training()

if __name__ == '__main__' and sys.argv[1] == 'predict':
    evaluate_one_image()

