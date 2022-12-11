import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser For Arguments', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-data', dest='dataset', default='FB15K-237', help='Dataset to use, default: FB15k-237')
    parser.add_argument('-folder', dest='folder', default='testdata', help='Folder to grab ground truths from')

    args = parser.parse_args()

    allfiles = sorted(os.listdir(f'./data/{args.dataset}/{args.folder}'), 
                key=lambda name: int(name.split('_timestep_')[1].split('_')[0]))

    change_points = []
    for i, file in enumerate(allfiles):
        change_percent = float(file.split('_')[3])
        if change_percent > 1.0:
            change_points.append(i+1)

    print("Change points are located at timesteps:")
    print(change_points)
