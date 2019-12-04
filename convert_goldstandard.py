import pandas as pd


def main():
    df = pd.read_csv('ref/SemEval2018-T3_gold_test_taskA_emoji.txt', encoding='utf8', sep='\t')
    df['Label'].to_csv('ref/goldstandard_train_A.txt', index=False)


if __name__ == '__main__':
    main()
