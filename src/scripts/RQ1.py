from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
import numpy as np
import argparse
import os
from scipy.stats import pearsonr
import sys

sys.path.append("..")
from scripts.load_movielens import load_movielens
from scripts.load_yelp import load_yelp
import influence.experiments as experiments
from influence.matrix_factorization import MF
from influence.NCF import NCF

configs = {
    "avextol": 1e-3,  # thresholdforoptimizationininfluencefunction
    "damping": 1e-6,  # dampingtermininfluencefunction
    "weight_decay": 1e-3,  # l2regularizationtermfortrainingMForNCFmodel
    "lr": 1e-3,  # initiallearningratefortrainingMForNCFmodel
    "embed_size": 16,  # embeddingsize
    "maxinf": 1,  # removetypeoftrainindices
    "dataset": "movielens",  # nameofdataset:movielensoryelp
    "model": "NCF",  # modeltype:MForNCF
    "num_test": 5,  # numberoftestpointsofretraining"
    "num_steps_train": 180000,  # trainingsteps
    "num_steps_retrain": 27000,  # retrainingsteps
    "reset_adam": 0,
    "load_checkpoint": 1,
    "retrain_times": 4,
    "sort_test_case": 0
}

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--avextol', type=float, default=1e-3,
#                         help='threshold for optimization in influence function')
#     parser.add_argument('--damping', type=float, default=1e-6,
#                         help='damping term in influence function')
#     parser.add_argument('--weight_decay', type=float, default=1e-3,
#                         help='l2 regularization term for training MF or NCF model')
#     parser.add_argument('--lr', type=float, default=1e-3,
#                         help='initial learning rate for training MF or NCF model')
#     parser.add_argument('--embed_size', type=int, default=16,
#                         help='embedding size')
#     parser.add_argument('--maxinf', type=int, default=1,
#                         help='remove type of train indices')
#     parser.add_argument('--dataset', type=str, default='movielens',
#                         help='name of dataset: movielens or yelp')
#     parser.add_argument('--model', type=str, default='NCF',
#                         help='model type: MF or NCF')
#     parser.add_argument('--num_test', type=int, default=5,
#                         help='number of test points of retraining')
#     parser.add_argument('--num_steps_train', type=int, default=180000,
#                         help='training steps')
#     parser.add_argument('--num_steps_retrain', type=int, default=27000,
#                         help='retraining steps')
#     parser.add_argument('--reset_adam', type=int, default=0)
#     parser.add_argument('--load_checkpoint', type=int, default=1)
#     parser.add_argument('--retrain_times', type=int, default=4)
#     parser.add_argument('--sort_test_case', type=int, default=0)
#     return parser.parse_args()

if configs['dataset'] == 'movielens':
    data_sets = load_movielens('../../data')
    batch_size = 3020
elif configs['dataset'] == 'yelp':
    data_sets = load_yelp('../../data')
    batch_size = 3009
else:
    raise NotImplementedError
weight_decay = configs['weight_decay']
initial_learning_rate = configs['lr']
num_users = int(np.max(data_sets["train"]._x[:, 0]) + 1)
num_items = int(np.max(data_sets["train"]._x[:, 1]) + 1)
print("number of users: %d" % num_users)
print("number of items: %d" % num_items)
print("number of training examples: %d" % data_sets["train"]._x.shape[0])
print("number of testing examples: %d" % data_sets["test"]._x.shape[0])
avextol = configs['avextol']
damping = configs['damping']
print("Using avextol of %.0e" % avextol)
print("Using damping of %.0e" % damping)
print("Using embedding size of %d" % configs['embed_size'])
if configs['model'] == 'MF':
    Model = MF
elif configs['model'] == 'NCF':
    Model = NCF
else:
    raise NotImplementedError

model = Model(
    num_users=num_users,
    num_items=num_items,
    embedding_size=configs['embed_size'],
    weight_decay=weight_decay,
    num_classes=1,
    batch_size=batch_size,
    data_sets=data_sets,
    initial_learning_rate=initial_learning_rate,
    damping=damping,
    decay_epochs=[10000, 20000],
    mini_batch=True,
    train_dir='output',
    log_dir='log',
    avextol=avextol,
    model_name='%s_%s_explicit_damping%.0e_avextol%.0e_embed%d_maxinf%d_wd%.0e' % (
        configs['dataset'], configs['model'], damping, avextol, configs['embed_size'], configs['maxinf'], weight_decay))
print(f'Model name is: {model.model_name}')

num_steps = configs['num_steps_train']
iter_to_load = num_steps - 1
if os.path.isfile("%s-%s.index" % (model.checkpoint_file, iter_to_load)):
    print('Checkpoint found, loading...')
    model.load_checkpoint(iter_to_load=iter_to_load)
else:
    print('Checkpoint not found, start training...')
    model.train(
        num_steps=num_steps)
    model.saver.save(model.sess, model.checkpoint_file, global_step=num_steps - 1)

if configs['maxinf']:
    remove_type = 'maxinf'
else:
    remove_type = 'random'

test_size = data_sets["test"].num_examples
num_test = configs['num_test']
test_indices = np.random.choice(test_size, num_test, replace=False)
if configs['sort_test_case']:
    num_related_ratings = []
    for i in range(test_size):
        num_related_ratings += [model.get_train_indices_of_test_case([i]).shape[0]]
    test_indices = np.argsort(np.array(num_related_ratings))[:num_test]

actual_y_diff = np.zeros(num_test)
predicted_y_diff = np.zeros(num_test)
removed_indices = np.zeros(num_test)

for i, test_idx in enumerate(test_indices):
    print(f'test point====={i}=====')
    actual_y_diffs, predicted_y_diffs, indices_to_remove = experiments.test_retraining(
        model,
        test_idx=test_idx,
        iter_to_load=iter_to_load,
        retrain_times=configs['retrain_times'],
        num_to_remove=1,
        num_steps=configs['num_steps_retrain'],
        remove_type=remove_type,
        force_refresh=True,
        reset_adam=configs['reset_adam'],
        load_checkpoint=configs['load_checkpoint'])
    actual_y_diff[i] = actual_y_diffs[0]
    predicted_y_diff[i] = predicted_y_diffs[0]
    removed_indices[i] = indices_to_remove[0]

np.savez(
    'output/RQ1-%s-%s.npz' % (configs['model'], configs['dataset']),
    actual_loss_diffs=actual_y_diff,
    predicted_loss_diffs=predicted_y_diff,
    indices_to_remove=removed_indices
)
print('Correlation is %s' % pearsonr(actual_y_diff, predicted_y_diff)[0])
