from helper.helper import *
from helper.data_loader import *
from tqdm import tqdm

# sys.path.append('./')
import pickle
from model.models import *

K = 100

class Runner(object):

    def construct_adj(self):
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

    def get_data_loader(self, dataset_class, split, batch_size, shuffle=False):
            return  DataLoader(
                    dataset_class(self.triples[split], self.p),
                    batch_size      = batch_size,
                    shuffle         = shuffle,
                    num_workers     = max(0, self.p.num_workers),
                    collate_fn      = dataset_class.collate_fn
                )

    def get_train_data(self, filename):
        self.data = ddict(list)
        sr2o = ddict(set)

        for line in open('./data/{}/{}/{}.txt'.format(self.p.dataset, self.p.trainfolder, filename)):
            sub, rel, obj = map(str.lower, line.strip().split('\t'))
            sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]
            self.data['train'].append((sub, rel, obj))

            sr2o[(sub, rel)].add(obj)
            sr2o[(obj, rel+self.p.num_rel)].add(sub)

        self.data = dict(self.data)

        self.sr2o = {k: list(v) for k, v in sr2o.items()}

        self.triples  = ddict(list)

        for (sub, rel), obj in self.sr2o.items():
            self.triples['train'].append({'triple':(sub, rel, -1), 'label': self.sr2o[(sub, rel)], 'sub_samp': 1})

        self.triples = dict(self.triples)

        self.data_iter = {
            'train':    	self.get_data_loader(TrainDataset, 'train', 	    256),
        }

        self.edge_index, self.edge_type = self.construct_adj()

    def load_init_data(self):
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

    def __init__(self, params):
        self.p			= params
        pprint(vars(self.p))

        if self.p.gpu != '-1' and torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.cuda.set_rng_state(torch.cuda.get_rng_state())
            torch.backends.cudnn.deterministic = True
        else:
            self.device = torch.device('cpu')

        self.load_init_data()

    def read_batch(self, batch, split):
        if split == 'train':
            triple, label = [ _.to(self.device) for _ in batch]
            return triple[:, 0], triple[:, 1], triple[:, 2], label
        else:
            triple, label = [ _.to(self.device) for _ in batch]
            return triple[:, 0], triple[:, 1], triple[:, 2], label

    def save_layer(self, save_path):
        state = {
            'state_dict'	: self.layer.state_dict(),
            'optimizer'	: self.optimizer.state_dict(),
            'args'		: vars(self.p)
        }
        torch.save(state, save_path)

    def load_layer(self, load_path):
        state			= torch.load(load_path, map_location=self.device)
        state_dict		= state['state_dict']

        self.layer.load_state_dict(state_dict)
        self.optimizer.load_state_dict(state['optimizer'])
    
    def load_model(self, load_path):
        state			= torch.load(load_path, map_location=self.device)
        state_dict		= state['state_dict']
        self.model.load_state_dict(state_dict)

    def train_siamese(self):
        save_path = os.path.join('./checkpoints', self.p.name)

        self.layer = torch.nn.Linear(in_features=K, out_features=1, bias=True).to(self.device)
        self.optimizer = torch.optim.SGD(self.layer.parameters(), lr=self.p.lr, momentum=0.9)

        # Load previous checkpoint
        if self.p.restore:
            self.load_layer(save_path)

        self.layer.train()

        # Using BCE with logits for better numerical stability
        criterion = torch.nn.BCEWithLogitsLoss().to(self.device)

        for epoch in range(self.p.max_epochs):
            # Sort the timesteps according to the timestep number -- IMP for training
            trainfiles = sorted(os.listdir(f'./data/{self.p.dataset}/{self.p.trainfolder}'), 
                key=lambda name: int(name.split('train_timestep_')[1].split('_')[0]))

            # Train only on first 200 files for faster speed
            trainfiles = trainfiles[:200]

            # Create pairs of tuples to propagate through the siamese network
            pairs = [(trainfiles[i], trainfiles[i + 1]) for i in range(len(trainfiles) - 1)]

            losses = []
            batches = None
            labels = []
            ent_emb1 = ddict()

            for i, (file1, file2) in enumerate(pairs):
                # Create label, if change point or not
                cp = float(file2.split('_')[3])
                label = torch.tensor([0.0]) if cp > 1.0 else torch.tensor([1.0])
                label = label.to(self.device)

                # Use previous step to avoid recomputation
                if len(ent_emb1) == 0:
                    ent_emb1 = self.get_embeddings(file1)

                ent_emb2 = self.get_embeddings(file2)
                intersecting_ents = list(set(ent_emb1.keys()) & set(ent_emb2.keys()))

                differences = [torch.linalg.norm(ent_emb1[k] - ent_emb2[k], ord=2) for k in intersecting_ents]
                differences = torch.topk(torch.tensor(differences).to(self.device), K)
                values = torch.unsqueeze(differences.values, dim=0)

                if batches is None:
                    batches = values
                    labels = label
                else:
                    batches = torch.cat((batches, values))
                    labels = torch.cat((labels, label))

                # Only do backward on batch size, till then accumulate
                if (i+1) % self.p.batch_size == 0:
                    # Pass through linear layer to train model
                    self.optimizer.zero_grad()
                    output = torch.squeeze(self.layer(batches))
                    loss = criterion(output, labels)
                    losses.append(loss.cpu().item())
                    loss.backward()
                    self.optimizer.step()

                    print('[Epoch:{}]: Loss:{:.4}'.format(epoch, loss.cpu().item()))

                    # Reset batch
                    batches = None

                print('[Pair: {}/{}]'.format(i, len(pairs)))
                # Important optimization, mark the second file as the first file for speedup
                ent_emb1 = ent_emb2

            print('[Epoch:{}]:  Training Loss:{:.4}'.format(epoch, np.mean(losses)))

            # Save model every epoch alternate epoch
            if (epoch+1) % 2 == 0:
                print("Saving model....")
                self.save_layer(save_path)
        
        self.save_layer(save_path)
                

    def get_embeddings(self, filename):
        # Create adjacency matrix for each train file
        self.get_train_data(os.path.splitext(filename)[0])
        self.model        = CompGCN_TransE(self.edge_index, self.edge_type, params=self.p)
        self.model.to(self.device)
        self.load_model(self.p.modelpath)
        self.model.eval()

        ent_embed = ddict()

        # Propagate training examples to gather embeddings for all entities
        with torch.no_grad():
            batch_iter = iter(self.data_iter['train'])
            for step, batch in tqdm(enumerate(batch_iter)):
                    sub, rel, _, _	= self.read_batch(batch, 'train')
                    _, sub_emb, _			= self.model.forward(sub, rel)

                    sub = sub.cpu().numpy()
                    sub_emb = sub_emb.cpu()

                    # Add the new embeddings to the global dict
                    ent_embed.update({int(sub[i]): e for i, e in enumerate(sub_emb)})
            
        return ent_embed

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser For Arguments', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-name',		default='testrun',					help='Set run name for saving/restoring models')
    parser.add_argument('-modelpath',		default='testrun',					help='Path of stored model to load and test')
    parser.add_argument('-data',		dest='dataset',         default='FB15K-237',            help='Dataset to use, default: FB15k-237')
    parser.add_argument('-trainfolder',		dest='trainfolder',         default='traindata',            help='Dataset to use, default: FB15k-237')
    parser.add_argument('-opn',             dest='opn',             default='sub',                 help='Composition Operation to be used in CompGCN')
    parser.add_argument('-gamma',		type=float,             default=40.0,			help='Margin')

    parser.add_argument('-batch',           dest='batch_size',      default=4,    type=int,       help='Batch size')
    parser.add_argument('-gpu',		type=str,               default='0',			help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
    parser.add_argument('-seed',            dest='seed',            default=41504,  type=int,     	help='Seed for randomization')
    parser.add_argument('-epoch',		dest='max_epochs', 	type=int,       default=10,  	help='Number of epochs')
    parser.add_argument('-l2',		type=float,             default=0.0,			help='L2 Regularization for Optimizer')
    parser.add_argument('-lr',		type=float,             default=0.001,			help='Starting Learning Rate')
    parser.add_argument('-num_workers',	type=int,               default=8,                     help='Number of processes to construct batches')
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

    args.restore = False
    # Only create new file when path exists otherwise resume training
    if not os.path.exists(os.path.join('./checkpoints', args.name)):
        args.name = args.name + '_' + time.strftime('%d_%m_%Y') + '_' + time.strftime('%H_%M_%S')
    else:
        args.restore = True

    model = Runner(args)
    model.train_siamese()
