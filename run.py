from helper.helper import *
from helper.data_loader import *

# sys.path.append('./')
from model.models import *

class Runner(object):

    def load_data(self):
        """
        Reading in raw triples and converts it into a standard format. 

        Parameters
        ----------
        self.p.dataset:         Takes in the name of the dataset (FB15k-237)
        
        Returns
        -------
        self.ent2id:            Entity to unique identifier mapping
        self.id2rel:            Inverse mapping of self.ent2id
        self.rel2id:            Relation to unique identifier mapping
        self.num_ent:           Number of entities in the Knowledge graph
        self.num_rel:           Number of relations in the Knowledge graph
        self.embed_dim:         Embedding dimension used
        self.data['train']:     Stores the triples corresponding to training dataset
        self.data['valid']:     Stores the triples corresponding to validation dataset
        self.data['test']:      Stores the triples corresponding to test dataset
        self.data_iter:		The dataloader for different data splits

        """

        ent_set, rel_set = OrderedSet(), OrderedSet()
        for split in ['train', 'test', 'valid']:
            for line in open('./data/{}/original_{}.txt'.format(self.p.dataset, split)):
                sub, rel, obj = map(str.lower, line.strip().split('\t'))
                ent_set.add(sub)
                rel_set.add(rel)
                ent_set.add(obj)

        self.ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
        self.rel2id = {rel: idx for idx, rel in enumerate(rel_set)}
        self.rel2id.update({rel+'_reverse': idx+len(self.rel2id) for idx, rel in enumerate(rel_set)})

        self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
        self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}

        self.p.num_ent		= len(self.ent2id)
        self.p.num_rel		= len(self.rel2id) // 2
        self.p.score_func   = 'transe'

        self.data = ddict(list)
        sr2o = ddict(set)

        for split in ['train', 'test', 'valid']:
            for line in open('./data/{}/{}.txt'.format(self.p.dataset, split)):
                sub, rel, obj = map(str.lower, line.strip().split('\t'))
                sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]
                self.data[split].append((sub, rel, obj))

                if split == 'train': 
                    sr2o[(sub, rel)].add(obj)
                    sr2o[(obj, rel+self.p.num_rel)].add(sub)

        self.data = dict(self.data)

        self.sr2o = {k: list(v) for k, v in sr2o.items()}
        for split in ['test', 'valid']:
            for sub, rel, obj in self.data[split]:
                sr2o[(sub, rel)].add(obj)
                sr2o[(obj, rel+self.p.num_rel)].add(sub)

        self.sr2o_all = {k: list(v) for k, v in sr2o.items()}
        self.triples  = ddict(list)

        for (sub, rel), obj in self.sr2o.items():
            self.triples['train'].append({'triple':(sub, rel, -1), 'label': self.sr2o[(sub, rel)], 'sub_samp': 1})

        for split in ['test', 'valid']:
            for sub, rel, obj in self.data[split]:
                rel_inv = rel + self.p.num_rel
                self.triples['{}_{}'.format(split, 'tail')].append({'triple': (sub, rel, obj), 	   'label': self.sr2o_all[(sub, rel)]})
                self.triples['{}_{}'.format(split, 'head')].append({'triple': (obj, rel_inv, sub), 'label': self.sr2o_all[(obj, rel_inv)]})

        self.triples = dict(self.triples)

        def get_data_loader(dataset_class, split, batch_size, shuffle=True):
            return  DataLoader(
                    dataset_class(self.triples[split], self.p),
                    batch_size      = batch_size,
                    shuffle         = shuffle,
                    num_workers     = max(0, self.p.num_workers),
                    collate_fn      = dataset_class.collate_fn
                )

        self.data_iter = {
            'train':    	get_data_loader(TrainDataset, 'train', 	    self.p.batch_size),
            'valid_head':   get_data_loader(TestDataset,  'valid_head', self.p.batch_size),
            'valid_tail':   get_data_loader(TestDataset,  'valid_tail', self.p.batch_size),
            'test_head':   	get_data_loader(TestDataset,  'test_head',  self.p.batch_size),
            'test_tail':   	get_data_loader(TestDataset,  'test_tail',  self.p.batch_size),
        }

        self.edge_index, self.edge_type = self.construct_adj()

    def construct_adj(self):
        """
        Constructor of the runner class

        Parameters
        ----------
        
        Returns
        -------
        Constructs the adjacency matrix for GCN
        
        """
        edge_index, edge_type = [], []

        for sub, rel, obj in self.data['train']:
            edge_index.append((sub, obj))
            edge_type.append(rel)

        # Adding inverse edges
        for sub, rel, obj in self.data['train']:
            edge_index.append((obj, sub))
            edge_type.append(rel + self.p.num_rel)

        edge_index	= torch.LongTensor(edge_index).to(self.device).t()
        edge_type	= torch.LongTensor(edge_type). to(self.device)

        return edge_index, edge_type

    def __init__(self, params):
        """
        Constructor of the runner class

        Parameters
        ----------
        params:         List of hyper-parameters of the model
        
        Returns
        -------
        Creates computational graph and optimizer
        
        """
        self.p			= params
        pprint(vars(self.p))

        if self.p.gpu != '-1' and torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.cuda.set_rng_state(torch.cuda.get_rng_state())
            torch.backends.cudnn.deterministic = True
        else:
            self.device = torch.device('cpu')

        self.load_data()
        self.model        = CompGCN_TransE(self.edge_index, self.edge_type, params=self.p)
        self.model.to(self.device)

        self.optimizer    = torch.optim.Adam(self.model.parameters(), lr=self.p.lr, weight_decay=self.p.l2)

    def read_batch(self, batch, split):
        """
        Function to read a batch of data and move the tensors in batch to CPU/GPU

        Parameters
        ----------
        batch: 		the batch to process
        split: (string) If split == 'train', 'valid' or 'test' split

        
        Returns
        -------
        Head, Relation, Tails, labels
        """
        if split == 'train':
            triple, label = [ _.to(self.device) for _ in batch]
            return triple[:, 0], triple[:, 1], triple[:, 2], label
        else:
            triple, label = [ _.to(self.device) for _ in batch]
            return triple[:, 0], triple[:, 1], triple[:, 2], label

    def save_model(self, save_path):
        """
        Function to save a model. It saves the model parameters, best validation scores,
        best epoch corresponding to best validation, state of the optimizer and all arguments for the run.

        Parameters
        ----------
        save_path: path where the model is saved
        
        Returns
        -------
        """
        state = {
            'state_dict'	: self.model.state_dict(),
            'best_val'	: self.best_val,
            'best_epoch'	: self.best_epoch,
            'optimizer'	: self.optimizer.state_dict(),
            'args'		: vars(self.p)
        }
        torch.save(state, save_path)

    def load_model(self, load_path):
        """
        Function to load a saved model

        Parameters
        ----------
        load_path: path to the saved model
        
        Returns
        -------
        """
        state			= torch.load(load_path)
        state_dict		= state['state_dict']
        self.best_val		= state['best_val']
        self.best_val_mrr	= self.best_val['mrr'] 

        self.model.load_state_dict(state_dict)
        self.optimizer.load_state_dict(state['optimizer'])

    def evaluate(self, split, epoch):
        """
        Function to evaluate the model on validation or test set

        Parameters
        ----------
        split: (string) If split == 'valid' then evaluate on the validation set, else the test set
        epoch: (int) Current epoch count
        
        Returns
        -------
        resutls:			The evaluation results containing the following:
            results['mr']:         	Average of ranks_left and ranks_right
            results['mrr']:         Mean Reciprocal Rank
            results['hits@k']:      Probability of getting the correct preodiction in top-k ranks based on predicted score

        """
        left_results  = self.predict(split=split, mode='tail_batch')
        right_results = self.predict(split=split, mode='head_batch')
        results       = get_combined_results(left_results, right_results)
        print('[Epoch {} {}]: MRR: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(epoch, split, results['left_mrr'], results['right_mrr'], results['mrr']))
        return results

    def predict(self, split='valid', mode='tail_batch'):
        """
        Function to run model evaluation for a given mode

        Parameters
        ----------
        split: (string) 	If split == 'valid' then evaluate on the validation set, else the test set
        mode: (string):		Can be 'head_batch' or 'tail_batch'
        
        Returns
        -------
        resutls:			The evaluation results containing the following:
            results['mr']:         	Average of ranks_left and ranks_right
            results['mrr']:         Mean Reciprocal Rank
            results['hits@k']:      Probability of getting the correct preodiction in top-k ranks based on predicted score

        """
        self.model.eval()

        with torch.no_grad():
            results = {}
            train_iter = iter(self.data_iter['{}_{}'.format(split, mode.split('_')[0])])

            for step, batch in enumerate(train_iter):
                sub, rel, obj, label	= self.read_batch(batch, split)
                pred, _, _			= self.model.forward(sub, rel)
                b_range			= torch.arange(pred.size()[0], device=self.device)
                target_pred		= pred[b_range, obj]
                pred 			= torch.where(label.byte(), -torch.ones_like(pred) * 10000000, pred)
                pred[b_range, obj] 	= target_pred
                ranks			= 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[b_range, obj]

                ranks 			= ranks.float()
                results['count']	= torch.numel(ranks) 		+ results.get('count', 0.0)
                results['mr']		= torch.sum(ranks).item() 	+ results.get('mr',    0.0)
                results['mrr']		= torch.sum(1.0/ranks).item()   + results.get('mrr',   0.0)
                for k in range(10):
                    results['hits@{}'.format(k+1)] = torch.numel(ranks[ranks <= (k+1)]) + results.get('hits@{}'.format(k+1), 0.0)

                if step % 100 == 0:
                    print('[{}, {} Step {}]\t{}'.format(split.title(), mode.title(), step, self.p.name))

        return results

    def run_epoch(self, epoch, val_mrr = 0):
        """
        Function to run one epoch of training

        Parameters
        ----------
        epoch: current epoch count
        
        Returns
        -------
        loss: The loss value after the completion of one epoch
        """
        self.model.train()
        losses = []
        train_iter = iter(self.data_iter['train'])

        for step, batch in enumerate(train_iter):
            self.optimizer.zero_grad()
            sub, rel, obj, label = self.read_batch(batch, 'train')

            pred, _, _	= self.model.forward(sub, rel)
            loss	= self.model.loss(pred, label)

            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

            if step % 100 == 0:
                print('[E:{}| {}]: Train Loss:{:.5},  Val MRR:{:.5}\t{}'.format(epoch, step, np.mean(losses), self.best_val_mrr, self.p.name))

        loss = np.mean(losses)
        print('[Epoch:{}]:  Training Loss:{:.4}\n'.format(epoch, loss))
        return loss

    def fit(self):
        """
        Function to run training and evaluation of model

        Parameters
        ----------
        
        Returns
        -------
        """
        self.best_val_mrr, self.best_val, self.best_epoch, val_mrr = 0., {}, 0, 0.
        save_path = os.path.join('./checkpoints', self.p.name)
        if args.restore:
            self.load_model(save_path)
            self.model.train()

        kill_cnt = 0
        for epoch in range(self.p.max_epochs):
            train_loss  = self.run_epoch(epoch, val_mrr)
            val_results = self.evaluate('valid', epoch)

            if val_results['mrr'] > self.best_val_mrr:
                self.best_val	   = val_results
                self.best_val_mrr  = val_results['mrr']
                self.best_epoch	   = epoch
                self.save_model(save_path)
                kill_cnt = 0
            else:
                kill_cnt += 1
                if kill_cnt % 10 == 0 and self.p.gamma > 5:
                    self.p.gamma -= 5 
                    print('Gamma decay on saturation, updated value of gamma: {}'.format(self.p.gamma))
                if kill_cnt > 25: 
                    print("Early Stopping!!")
                    break

            print('[Epoch {}]: Training Loss: {:.5}, Valid MRR: {:.5}\n\n'.format(epoch, train_loss, self.best_val_mrr))

        print('Loading best model, Evaluating on Test data')
        self.load_model(save_path)
        test_results = self.evaluate('test', epoch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser For Arguments', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-name',		default='testrun',					help='Set run name for saving/restoring models')
    parser.add_argument('-data',		dest='dataset',         default='FB15K-237',            help='Dataset to use, default: FB15k-237')
    parser.add_argument('-model',		dest='model',		default='compgcn',		help='Model Name')
    parser.add_argument('-opn',             dest='opn',             default='sub',                 help='Composition Operation to be used in CompGCN')

    parser.add_argument('-batch',           dest='batch_size',      default=128,    type=int,       help='Batch size')
    parser.add_argument('-gamma',		type=float,             default=40.0,			help='Margin')
    parser.add_argument('-gpu',		type=str,               default='0',			help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
    parser.add_argument('-epoch',		dest='max_epochs', 	type=int,       default=500,  	help='Number of epochs')
    parser.add_argument('-l2',		type=float,             default=0.0,			help='L2 Regularization for Optimizer')
    parser.add_argument('-lr',		type=float,             default=0.001,			help='Starting Learning Rate')
    parser.add_argument('-num_workers',	type=int,               default=10,                     help='Number of processes to construct batches')
    parser.add_argument('-seed',            dest='seed',            default=41504,  type=int,     	help='Seed for randomization')
    parser.add_argument('-bias',            dest='bias',            action='store_true',            help='Whether to use bias in the model')
    parser.add_argument('-num_bases',	dest='num_bases', 	default=-1,   	type=int, 	help='Number of basis relation vectors to use')
    parser.add_argument('-init_dim',	dest='init_dim',	default=100,	type=int,	help='Initial dimension size for entities and relations')
    parser.add_argument('-gcn_dim',	  	dest='gcn_dim', 	default=200,   	type=int, 	help='Number of hidden units in GCN')
    parser.add_argument('-embed_dim',	dest='embed_dim', 	default=None,   type=int, 	help='Embedding dimension to give as input to score function')
    parser.add_argument('-gcn_layer',	dest='gcn_layer', 	default=1,   	type=int, 	help='Number of GCN Layers to use')
    parser.add_argument('-gcn_drop',	dest='dropout', 	default=0.1,  	type=float,	help='Dropout to use in GCN Layer')
    parser.add_argument('-hid_drop',  	dest='hid_drop', 	default=0.3,  	type=float,	help='Dropout after GCN')

    args = parser.parse_args()

    args.restore = False
    # Only create new file when path exists otherwise resume training
    if not os.path.exists(os.path.join('./checkpoints', args.name)):
        args.name = args.name + '_' + time.strftime('%d_%m_%Y') + '_' + time.strftime('%H_%M_%S')
    else:
        args.restore = True

    set_gpu(args.gpu)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Set GPU seed
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(args.seed)

    model = Runner(args)
    model.fit()
