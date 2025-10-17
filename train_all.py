from pretrain import main_pretrain
from classifier import main_train_classifier

if __name__ == '__main__':
    for i in range(0, 10):
        print(f'Run {i}')
        main_pretrain(i)
        main_train_classifier(i)
