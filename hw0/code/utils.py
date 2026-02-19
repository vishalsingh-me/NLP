# CS505: NLP - Spring 2026

def calculate_accuracy(predictions, labels):
    if len(predictions) == 0:
        return 0.0
    
    correct = sum(1 for p, l in zip(predictions, labels) if p == l)
    return correct / len(predictions)

def macro_f1(predictions, labels, num_classes=4):
    # TODO: implement the macro-F1 score.
    # Recall that this involves computing the F1 score separately for
    # each label, and then taking the macroaverage. Return the macro-F1
    # score as a floating-point number.
    # STUDENT START --------------------------------------
    if len(predictions) == 0:
        return 0.0

    f1_scores = []
    for cls in range(num_classes):
        tp = sum(1 for p, l in zip(predictions, labels) if p == cls and l == cls)
        fp = sum(1 for p, l in zip(predictions, labels) if p == cls and l != cls)
        fn = sum(1 for p, l in zip(predictions, labels) if p != cls and l == cls)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)

    return sum(f1_scores) / num_classes if num_classes > 0 else 0.0
    # STUDENT END -------------------------------------------
