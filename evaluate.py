
if __name__ == "__main__":
    GT = "ref/SemEval2018-T3_gold_test_taskA_emoji.txt"
    Res = "res/predictions-taskA.txt"

    gt = []
    res = []

    with open(GT) as f:
        tmp = f.readlines()

        for i in range(1,len(tmp)):
            l = tmp[i]
            l = l.split('\t')
            gt.append(int(l[1]))

    with open(Res) as f:
        tmp = f.readlines()
        for i in range(len(tmp)):
            res.append(int(tmp[i]))

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    if(len(res) != len(gt)):
        print("Length of output is not right!!!")
    else:
        print("Length is ok!")

    for i in range(len(res)):
        if(res[i] == 1):
            if(gt[i] == 1):
                tp += 1
            else:
                fp += 1
        else:
            if(gt[i] == 1):
                fn += 1
            else:
                tn += 1

    accuracy = float(tp + tn) / len(res)
    print("accuracy = ", accuracy)

    precision = float(tp) / (tp + fp)
    print("precision = ", precision)

    recall = float(tp) / (tp + fn)
    print("recall = ", recall)

    f1 = 2.0 * (precision*recall) / (precision + recall)
    print("F1 = ", f1)
