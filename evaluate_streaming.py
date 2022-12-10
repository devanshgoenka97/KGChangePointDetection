import sys
import os
from collections import defaultdict
from evaluate import Runner
from helper.helper import *
from helper.data_loader import *

# Run the model on the streaming test set to get the error
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser For Arguments', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-modelpath',		default='testrun',					help='Path of stored model to load and test')
    parser.add_argument('-data',		dest='dataset',         default='FB15K-237',            help='Dataset to use, default: FB15k-237')
    parser.add_argument('-testfolder',  dest='testfolder',       default='testdata',             help='Folder where snapshot test data is stored')
    parser.add_argument('-opn',             dest='opn',             default='sub',                 help='Composition Operation to be used in CompGCN')
    parser.add_argument('-gamma',		type=float,             default=40.0,			help='Margin')

    parser.add_argument('-batch',           dest='batch_size',      default=128,    type=int,       help='Batch size')
    parser.add_argument('-gpu',		type=str,               default='0',			help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
    parser.add_argument('-seed',            dest='seed',            default=41504,  type=int,     	help='Seed for randomization')
    parser.add_argument('-epoch',		dest='max_epochs', 	type=int,       default=500,  	help='Number of epochs')
    parser.add_argument('-l2',		type=float,             default=0.0,			help='L2 Regularization for Optimizer')
    parser.add_argument('-lr',		type=float,             default=0.001,			help='Starting Learning Rate')
    parser.add_argument('-num_workers',	type=int,               default=10,                     help='Number of processes to construct batches')
    parser.add_argument('-bias',            dest='bias',            action='store_true',            help='Whether to use bias in the model')
    parser.add_argument('-num_bases',	dest='num_bases', 	default=-1,   	type=int, 	help='Number of basis relation vectors to use')
    parser.add_argument('-init_dim',	dest='init_dim',	default=100,	type=int,	help='Initial dimension size for entities and relations')
    parser.add_argument('-gcn_dim',	  	dest='gcn_dim', 	default=200,   	type=int, 	help='Number of hidden units in GCN')
    parser.add_argument('-embed_dim',	dest='embed_dim', 	default=None,   type=int, 	help='Embedding dimension to give as input to score function')
    parser.add_argument('-gcn_layer',	dest='gcn_layer', 	default=1,   	type=int, 	help='Number of GCN Layers to use')
    parser.add_argument('-gcn_drop',	dest='dropout', 	default=0.1,  	type=float,	help='Dropout to use in GCN Layer')
    parser.add_argument('-hid_drop',  	dest='hid_drop', 	default=0.3,  	type=float,	help='Dropout after GCN')

    args = parser.parse_args()

    set_gpu(args.gpu)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Set GPU seed
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(args.seed)

    results = defaultdict()

    for file_ in os.listdir(f'./data/{args.dataset}/{args.testfolder}'):
        file_ = os.path.splitext(file_)[0]
        timestep = int(file_.split('timestep_')[1].split('_')[0])
        print(f'Processing file: {file_}')
        args.testfilename = args.testfolder + '/' + file_
        model = Runner(args)
        result = model.evaluate('test', 100)
        results[timestep] = result

    results = sorted(results.items())
    f = open(f'./stream_mrr_{args.dataset}.txt', 'a')

    for k, v in results:
        print(f"MRR for timestep {k} = {v['mrr']}")
        f.write(f"{v['mrr']}\n")

    f.close()
    print(f"Done writing results to stream_mrr_{args.dataset}.txt")
        

