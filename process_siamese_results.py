import argparse

def compute_avg_delay(ground_truths, predictions):
    total_delay = 0
    for prediction in predictions:
        min_delay = None
        for break_point in ground_truths:
            delay = abs(break_point - prediction)
            if min_delay == None or min_delay > delay:
                min_delay = delay
        total_delay += min_delay
    avg_delay = total_delay / (len(ground_truths))
    return total_delay, avg_delay

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser For Arguments', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-path', default='results/siamese_FB15K-237_testdata3')
    parser.add_argument('-threshold', default='0.40')

    args = parser.parse_args()

    THRESHOLD = float(args.threshold)

    f = open(args.path, 'r')

    change_points = []
    predictions = []
    hit_rate = 0
    for ts, line in enumerate(f):
        score = float(line.split('\t')[0])
        expected = int(line.split('\t')[1])
        if expected == 0:
            change_points.append(ts+1)

        if score < THRESHOLD:
            predictions.append(ts+1)

        if expected == 0 and score < THRESHOLD:
            hit_rate += 1

    total, avg = compute_avg_delay(change_points, predictions)
    print("Total Delay = " + str(total))
    print("Avg Delay = " + str(avg))
    print("Hit Rate = " + str(100*hit_rate/len(change_points)))
    print("Total predictions = " + str(len(predictions)))
    print("Total ground truth = " + str(len(change_points)))
