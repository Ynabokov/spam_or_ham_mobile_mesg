def metrics(training_size, labels, predictions):
    true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0
    for i in range(len(labels)):
        true_pos += int(labels[i] == 1 and predictions[i] == 1)
        true_neg += int(labels[i] == 0 and predictions[i] == 0)
        false_pos += int(labels[i] == 0 and predictions[i] == 1)
        false_neg += int(labels[i] == 1 and predictions[i] == 0)
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    Fscore = 2 * precision * recall / (precision + recall)
    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)

    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F-score: ", Fscore)
    print("Accuracy: ", accuracy)

def evaluate():

    reference = []
    bigram = []
    unigram = []
    training_size = 0
    with open("data/reference/ngram.out", "r") as reference, open("output/bigram.out", "r") as bigram, open("output/unigram.out", "r") as unigram:
        reference = list(map(int, reference.readline().split(",")))
        bigram = list(map(int, bigram.readline().split(",")))
        unigram = list(map(int, unigram.readline().split(",")))

    print("Bigram results:")
    metrics(training_size, reference, bigram)
    print("")
    print("Unigram results:")
    metrics(training_size, reference, unigram)

if __name__ == "__main__":
    evaluate()

 
