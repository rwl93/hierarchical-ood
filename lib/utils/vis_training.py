import argparse
import matplotlib.pyplot as plt
import re


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--train-logs', nargs='+', metavar="FILENAME",
                        help='Train logs to plot')
    parser.add_argument('-s', '--save',
                        action='store_true',)
    parser.add_argument('--outfile', metavar="FILENAME",
                        default='train_logs.png',
                        help='Log plot output file')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    steps = []
    losses = []
    for log in args.train_logs:
        with open(log, 'r') as f:
            lines = f.read().splitlines()
        ss = []
        ls = []
        for l in lines:
            if "Step" in l:
                ss.append(int(re.search('(?<=Step=)\d+', l).group()))
                ls.append(float(re.search('(?<=Loss=)[.\d]+', l).group()))
        steps.append(ss)
        losses.append(ls)

    fig = plt.figure()
    plt.ylabel('Loss')
    plt.xlabel('Steps')
    legend = []
    for i, (s, l) in enumerate(zip(steps, losses)):
        plt.plot(s,l)
        legend.append(args.train_logs[i].split('/')[-2])
    plt.legend(legend)
    if args.save:
        pass
    plt.show()


if __name__ == "__main__":
    main()
