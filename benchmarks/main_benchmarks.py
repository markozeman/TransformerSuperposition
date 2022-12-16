import os
import sys
import argparse
import torch
import time
import numpy as np
from random import shuffle
from collections import OrderedDict
from torch import Tensor
from torch.utils.data import TensorDataset
import agents
from agents.default import plot_multiple_histograms
from Split_CIFAR_100_preparation import get_dataset


def get_data(s):
    """
    Get data for string s abbreviation.

    :param s: string of dataset abbreviation name
    :return: X, y, mask
    """
    if s == 'HS':
        X = torch.load('../Word2Vec_embeddings/X_hate_speech.pt').float()
        y = torch.load('../Word2Vec_embeddings/y_hate_speech.pt')
        mask = torch.load('../Word2Vec_embeddings/mask_hate_speech.pt').float()
    elif s == 'SA':
        X = torch.load('../Word2Vec_embeddings/X_IMDB_sentiment_analysis.pt').float()
        y = torch.load('../Word2Vec_embeddings/y_IMDB_sentiment_analysis.pt')
        mask = torch.load('../Word2Vec_embeddings/mask_IMDB_sentiment_analysis.pt').float()
    elif s == 'S':
        X = torch.load('../Word2Vec_embeddings/X_sms_spam.pt').float()
        y = torch.load('../Word2Vec_embeddings/y_sms_spam.pt')
        mask = torch.load('../Word2Vec_embeddings/mask_sms_spam.pt').float()
    elif s == 'SA_2':
        X = torch.load('../Word2Vec_embeddings/X_sentiment_analysis_2.pt').float()
        y = torch.load('../Word2Vec_embeddings/y_sentiment_analysis_2.pt')
        mask = torch.load('../Word2Vec_embeddings/mask_sentiment_analysis_2.pt').float()
    elif s == 'C':
        X = torch.load('../Word2Vec_embeddings/X_clickbait.pt').float()
        y = torch.load('../Word2Vec_embeddings/y_clickbait.pt')
        mask = torch.load('../Word2Vec_embeddings/mask_clickbait.pt').float()
    elif s == 'HD':
        X = torch.load('../Word2Vec_embeddings/X_humor_detection.pt').float()
        y = torch.load('../Word2Vec_embeddings/y_humor_detection.pt')
        mask = torch.load('../Word2Vec_embeddings/mask_humor_detection.pt').float()

    return X, y, mask


def run(args, do_early_stopping, stopping_criteria, task_names, num_tasks, times_per_task=None, previous_time=None, split_cifar_100=None):
    if not os.path.exists('outputs'):
        os.mkdir('outputs')

    task_output_space = {name: 2 for name in task_names} if args.network_type == 'transformer' else {name: 10 for name in task_names}
    test_all_tasks = []

    # Prepare the Agent (model)
    agent_config = {'lr': args.lr, 'momentum': args.momentum, 'weight_decay': args.weight_decay,'schedule': args.schedule,
                    'model_type':args.model_type, 'model_name': args.model_name, 'model_weights':args.model_weights,
                    'out_dim':{'All':args.force_out_dim} if args.force_out_dim>0 else task_output_space,
                    'optimizer':args.optimizer,
                    'print_freq':args.print_freq, 'gpuid': args.gpuid,
                    'reg_coef':args.reg_coef}
    agent = agents.__dict__[args.agent_type].__dict__[args.agent_name](agent_config)
    print(agent.model)
    print('#parameter of model:',agent.count_parameter())

    # Decide split ordering
    # task_names = sorted(list(task_output_space.keys()), key=int)
    # print('Task order:',task_names)
    # if args.rand_split_order:
    #     shuffle(task_names)
    #     print('Shuffled task order:', task_names)

    acc_table = OrderedDict()
    auroc_table = OrderedDict()
    auprc_table = OrderedDict()
    epochs = []

    if args.offline_training:  # Non-incremental learning / offline_training / measure the upper-bound performance
        pass

    else:  # Incremental learning
        # Feed data to agent and evaluate agent's performance
        for i in range(num_tasks):
            train_name = task_names[i]
            print('======================',train_name,'=======================')

            # prepare data
            if args.network_type == 'transformer':
                X, y, mask = get_data(train_name)

                # split data into train, validation and test set
                y = torch.max(y, 1)[1]  # change one-hot-encoded vectors to numbers
                permutation = torch.randperm(X.size()[0])
                X = X[permutation]
                y = y[permutation]
                mask = mask[permutation]
                index_val = round(0.8 * len(permutation))
                index_test = round(0.9 * len(permutation))

                # X_train, y_train, mask_train = X[:index_val, :, :].to("cuda:0"), y[:index_val].to("cuda:0"), mask[:index_val,:].to("cuda:0")  # train data to GPU
                X_train, y_train, mask_train = X[:index_val, :, :], y[:index_val], mask[:index_val,:]
                X_val, y_val, mask_val = X[index_val:index_test, :, :], y[index_val:index_test], mask[index_val:index_test,:]
                X_test, y_test, mask_test = X[index_test:, :, :], y[index_test:], mask[index_test:, :]
            elif args.network_type == 'MLP' or args.network_type == 'CNN':
                X_train, y_train, X_test, y_test = split_cifar_100[i]
                y_train = torch.max(y_train, 1)[1]  # change one-hot-encoded vectors to numbers
                y_test = torch.max(y_test, 1)[1]  # change one-hot-encoded vectors to numbers
                permutation = torch.randperm(X_train.size()[0])
                X_train = X_train[permutation]
                y_train = y_train[permutation]

                mask_train_val = torch.FloatTensor([0] * len(X_train))  # only for the purpose of compatibility with transformer network
                mask_test = torch.FloatTensor([0] * len(X_test))  # only for the purpose of compatibility with transformer network
                index_val = round(0.9 * len(permutation))  # use 10% of test set as validation set

                if args.network_type == 'MLP':
                    X_val, y_val, mask_val = X_train[index_val:, :], y_train[index_val:], mask_train_val[index_val:]
                    X_train, y_train, mask_train = X_train[:index_val, :], y_train[:index_val], mask_train_val[:index_val]

                elif args.network_type == 'CNN':
                    X_val, y_val, mask_val = X_train[index_val:, :, :, :], y_train[index_val:], mask_train_val[index_val:]
                    X_train, y_train, mask_train = X_train[:index_val, :, :, :], y_train[:index_val], mask_train_val[:index_val]

            # tasks4compatibility = np.array(tuple([str(i + 1)] * args.batch_size))
            train_dataset = TensorDataset(X_train, y_train, mask_train)     # mask not used
            val_dataset = TensorDataset(X_val, y_val, mask_val)             # mask not used
            test_dataset = TensorDataset(X_test, y_test, mask_test)         # mask not used

            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)

            # save test data for every task
            test_all_tasks.append(test_loader)

            if args.incremental_class:
                agent.add_valid_output_dim(task_output_space[train_name])

            # Learn
            epochs_trained = agent.learn_batch(train_loader, val_loader, do_early_stopping, stopping_criteria)
            epochs.append(epochs_trained)

            # Evaluate
            acc_table[train_name] = OrderedDict()
            auroc_table[train_name] = OrderedDict()
            auprc_table[train_name] = OrderedDict()
            for j in range(i+1):
                val_name = task_names[j]
                print('Test split name:', val_name)

                acc, auroc, auprc = agent.validation(test_all_tasks[j], is_test=True)
                print('acc, auroc, auprc:', acc, auroc, auprc)

                acc_table[val_name][train_name] = acc
                auroc_table[val_name][train_name] = auroc
                auprc_table[val_name][train_name] = auprc

            if times_per_task is not None:
                print('Elapsed time so far: %.2f s, %.2f min' % (time.time() - previous_time, (time.time() - previous_time) / 60))
                times_per_task[r, i] = time.time() - previous_time  # saved in seconds
                previous_time = time.time()

    return acc_table, auroc_table, auprc_table, task_names, epochs, times_per_task


def get_args(argv):
    # This function prepares the variables shared across demo.py
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0],
                        help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
    parser.add_argument('--model_type', type=str, default='mlp', help="The type (mlp|lenet|vgg|resnet) of backbone network")
    parser.add_argument('--model_name', type=str, default='MLP', help="The name of actual model for the backbone")
    parser.add_argument('--force_out_dim', type=int, default=2, help="Set 0 to let the task decide the required output dimension")
    parser.add_argument('--agent_type', type=str, default='default', help="The type (filename) of agent")
    parser.add_argument('--agent_name', type=str, default='NormalNN', help="The class name of agent")
    parser.add_argument('--optimizer', type=str, default='Adam', help="SGD|Adam|RMSprop|amsgrad|Adadelta|Adagrad|Adamax ...")
    parser.add_argument('--dataroot', type=str, default='data', help="The root folder of dataset or downloaded data")
    parser.add_argument('--dataset', type=str, default='MNIST', help="MNIST(default)|CIFAR10|CIFAR100")
    parser.add_argument('--n_permutation', type=int, default=10, help="Enable permuted tests when >0")
    parser.add_argument('--first_split_size', type=int, default=2)
    parser.add_argument('--other_split_size', type=int, default=2)
    parser.add_argument('--no_class_remap', dest='no_class_remap', default=True, action='store_true',
                        help="Avoid the dataset with a subset of classes doing the remapping. Ex: [2,5,6 ...] -> [0,1,2 ...]")
    parser.add_argument('--train_aug', dest='train_aug', default=False, action='store_true',
                        help="Allow data augmentation during training")
    parser.add_argument('--rand_split', dest='rand_split', default=False, action='store_true',
                        help="Randomize the classes in splits")
    parser.add_argument('--rand_split_order', dest='rand_split_order', default=False, action='store_true',
                        help="Randomize the order of splits")
    parser.add_argument('--workers', type=int, default=3, help="#Thread for dataloader")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--momentum', type=float, default=0)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--schedule', nargs="+", type=int, default=[2],
                        help="The list of epoch numbers to reduce learning rate by factor of 0.1. Last number is the end epoch")
    parser.add_argument('--print_freq', type=float, default=100, help="Print the log at every x iteration")
    parser.add_argument('--model_weights', type=str, default=None,
                        help="The path to the file for the model weights (*.pth).")
    parser.add_argument('--reg_coef', nargs="+", type=float, default=[0.], help="The coefficient for regularization. Larger means less plasilicity. Give a list for hyperparameter search.")
    parser.add_argument('--eval_on_train_set', dest='eval_on_train_set', default=False, action='store_true',
                        help="Force the evaluation on train set")
    parser.add_argument('--offline_training', dest='offline_training', default=False, action='store_true',
                        help="Non-incremental learning by make all data available in one batch. For measuring the upperbound performance.")
    parser.add_argument('--repeat', type=int, default=1, help="Repeat the experiment N times")
    parser.add_argument('--incremental_class', dest='incremental_class', default=False, action='store_true',
                        help="The number of output node in the single-headed model increases along with new categories.")
    parser.add_argument('--method', type=str, default='EWC', choices=['EWC', 'Online_EWC', 'SI', 'MAS', 'GEM_Large', 'GEM_Small'])
    parser.add_argument('--network_type', type=str, default='transformer', choices=['transformer', 'MLP', 'CNN'])
    parser.add_argument('--num_runs', type=int, default=5)
    args = parser.parse_args(argv)
    return args


if __name__ == '__main__':
    args = get_args(sys.argv[1:])
    reg_coef_list = args.reg_coef
    avg_final_acc = {}
    do_early_stopping = True
    stopping_criteria = 'auroc' if args.network_type == 'transformer' else 'acc'  # possibilities: 'acc', 'auroc', 'auprc'
    args.repeat = args.num_runs   # number of runs

    if args.method == 'EWC':
        args.method = 'EWC_mnist'
    elif args.method == 'Online_EWC':
        args.method = 'EWC_online_mnist'
    elif args.method == 'GEM_Large':
        args.method = 'GEM_300' if args.network_type == 'transformer' else 'GEM_500'    # 50 memory slots per task
    elif args.method == 'GEM_Small':
        args.method = 'GEM_30' if args.network_type == 'transformer' else 'GEM_50'      # 5 memory slots per task
    args.agent_name = args.method   # continual learning method; options: ['EWC_mnist', 'EWC_online_mnist', 'SI', 'MAS', 'GEM_x']

    args.force_out_dim = 2 if args.network_type == 'transformer' else 10  # number of output neurons / number of classes
    num_tasks = 6 if args.network_type == 'transformer' else 10
    args.lr = 0.0001 if args.network_type == 'CNN' else 0.001

    if args.network_type == 'transformer':
        args.model_name = 'myTransformer'  # to use Transformer model
    elif args.network_type == 'MLP':
        args.model_name = 'myMLP'
    elif args.network_type == 'CNN':
        args.model_name = 'myCNN'

    # load Split CIFAR-100
    if args.network_type == 'MLP':
        split_cifar_100 = get_dataset('nn', (32, 32, 3))
    elif args.network_type == 'CNN':
        split_cifar_100 = get_dataset('cnn', (32, 32, 3))
    else:
        split_cifar_100 = None

    args.agent_type = 'regularization' if args.agent_name == 'MAS' or args.agent_name == 'SI' else 'customization'
    # args.optimizer = 'SGD' if args.agent_name.startswith('GEM') else 'Adam'
    best_reg_coefs = {      # based on coefficient search
        'EWC_mnist': 5000,  # _mnist does not mean we run it on MNIST (used for backward compatibility)
        'EWC_online_mnist': 5000,   # _mnist does not mean we run it on MNIST (used for backward compatibility)
        'SI': 2,
        'MAS': 50,
        'GEM_510': 0.005,
        'GEM_500': 0.005,
        'GEM_300': 0.005,
        'GEM_60': 0.005,
        'GEM_50': 0.005,
        'GEM_30': 0.005,
        'GEM_15': 0.005,
        'GEM_12': 0.005
    }
    reg_coef_list = [best_reg_coefs[args.agent_name]]

    # The for loops over hyper-paramerters or repeats
    for reg_coef in reg_coef_list:
        args.reg_coef = reg_coef
        avg_final_acc[reg_coef] = np.zeros(args.repeat)

        avg_acc_history_all_repeats = []
        avg_auroc_history_all_repeats = []
        avg_auprc_history_all_repeats = []
        times_per_run = []
        times_per_task = np.zeros((args.repeat, num_tasks))
        epochs_per_run = []

        if args.network_type == 'transformer':
            runs_task_names = [['HS', 'SA', 'S', 'SA_2', 'C', 'HD'],
                               ['C', 'HD', 'SA', 'HS', 'SA_2', 'S'],
                               ['SA', 'S', 'HS', 'SA_2', 'HD', 'C'],
                               ['HD', 'SA_2', 'SA', 'C', 'S', 'HS'],
                               ['SA', 'HS', 'C', 'SA_2', 'HD', 'S']]
        else:   # MLP or CNN - use Split CIFAR-100
            runs_task_names = [list(map(str, list(range(num_tasks))))] * args.num_runs

        for r in range(args.repeat):
            print('- - Run %d - -' % (r + 1))

            start_time = time.time()
            previous_time = start_time

            # Run the experiment
            acc_table, auroc_table, auprc_table, task_names, epochs, times_per_task = \
                run(args, do_early_stopping, stopping_criteria, runs_task_names[r], num_tasks, times_per_task, previous_time, split_cifar_100)
            print('Accuracy dict:', acc_table)
            print('AUROC dict:', auroc_table)
            print('AUPRC dict:', auprc_table)

            epochs_per_run.append(epochs)

            # Calculate average performance across tasks
            # Customize this part for a different performance metric
            avg_acc_history = [0] * num_tasks
            avg_auroc_history = [0] * num_tasks
            avg_auprc_history = [0] * num_tasks
            for i in range(num_tasks):
                train_name = task_names[i]
                cls_acc_sum = 0
                cls_auroc_sum = 0
                cls_auprc_sum = 0
                for j in range(i + 1):
                    val_name = task_names[j]
                    cls_acc_sum += acc_table[val_name][train_name]
                    cls_auroc_sum += auroc_table[val_name][train_name]
                    cls_auprc_sum += auprc_table[val_name][train_name]

                avg_acc_history[i] = cls_acc_sum / (i + 1)
                avg_auroc_history[i] = cls_auroc_sum / (i + 1)
                avg_auprc_history[i] = cls_auprc_sum / (i + 1)
                print('\nAverage test accuracy until task', train_name, ':', avg_acc_history[i])
                print('Average test AUROC until task', train_name, ':', avg_auroc_history[i])
                print('Average test AUPRC until task', train_name, ':', avg_auprc_history[i])

            avg_acc_history_all_repeats.append(avg_acc_history)
            avg_auroc_history_all_repeats.append(avg_auroc_history)
            avg_auprc_history_all_repeats.append(avg_auprc_history)

            # Gather the final avg accuracy
            avg_final_acc[reg_coef][r] = avg_acc_history[-1]

            # Print the summary so far
            print('===Summary of experiment repeats:',r+1,'/',args.repeat,'===')
            print('The regularization coefficient:', args.reg_coef)
            print('The last avg acc of all repeats:', avg_final_acc[reg_coef])
            print('mean:', avg_final_acc[reg_coef].mean(), 'std:', avg_final_acc[reg_coef].std())

            end_time = time.time()
            time_elapsed = end_time - start_time
            times_per_run.append(time_elapsed)
            print('Time elapsed for this run:', round(time_elapsed, 2), 's')

        print('\nEpochs per run: ', epochs_per_run)
        print('Times per run: ', times_per_run)
        print('Runs: %d,  Average time per run: %.2f +/- %.2f s, %.1f +/- %.1f min' %
              (args.repeat, np.mean(np.array(times_per_run)), np.std(np.array(times_per_run)), np.mean(np.array(times_per_run)) / 60, np.std(np.array(times_per_run)) / 60))
        print('Runs: %d,  Average #epochs for all tasks: %.2f +/- %.2f' %
              (args.repeat, np.mean(np.array([sum(l) for l in epochs_per_run])), np.std(np.array([sum(l) for l in epochs_per_run]))))

        avg_acc_history_all_repeats = np.array(avg_acc_history_all_repeats)
        avg_auroc_history_all_repeats = np.array(avg_auroc_history_all_repeats)
        avg_auprc_history_all_repeats = np.array(avg_auprc_history_all_repeats)
        print('\nAccuracy after all repeats: ', avg_acc_history_all_repeats)
        print('AUROC after all repeats: ', avg_auroc_history_all_repeats)
        print('AUPRC after all repeats: ', avg_auprc_history_all_repeats)

    for reg_coef,v in avg_final_acc.items():
        print('\nreg_coef:', reg_coef,'mean:', avg_final_acc[reg_coef].mean(), 'std:', avg_final_acc[reg_coef].std())


    ### Plot values of acc, auroc and auprc for average values until every task
    # num_tasks = len(task_names)
    min_y = 30 if args.network_type == 'transformer' else 0
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    mean_acc, std_acc = np.mean(avg_acc_history_all_repeats, axis=0), np.std(avg_acc_history_all_repeats, axis=0)
    mean_auroc, std_auroc = np.mean(avg_auroc_history_all_repeats, axis=0), np.std(avg_auroc_history_all_repeats, axis=0)
    mean_auprc, std_auprc = np.mean(avg_auprc_history_all_repeats, axis=0), np.std(avg_auprc_history_all_repeats, axis=0)

    mean_time_per_task, std_time_per_task = np.mean(times_per_task, axis=0), np.std(times_per_task, axis=0)

    print('\nPerformance means until a specific task:')
    for t in range(num_tasks):
        print('------------------------------------------')
        print('Mean time for task %d: %.2f +/- %.2f s,  %.2f +/- %.2f min' % (t+1, mean_time_per_task[t], std_time_per_task[t], mean_time_per_task[t] / 60, std_time_per_task[t] / 60))
        print('Mean time until task %d: %.2f s,  %.2f min' % (t+1, sum(mean_time_per_task[:t + 1]), sum(mean_time_per_task[:t + 1]) / 60))
        print('Task %d - Accuracy = %.1f +/- %.1f' % (t + 1, mean_acc[t], std_acc[t]))
        print('Task %d - AUROC    = %.1f +/- %.1f' % (t + 1, mean_auroc[t], std_auroc[t]))
        print('Task %d - AUPRC    = %.1f +/- %.1f' % (t + 1, mean_auprc[t], std_auprc[t]))

    end_performance = {i: {'acc': 0, 'auroc': 0, 'auprc': 0, 'std_acc': 0, 'std_auroc': 0, 'std_auprc': 0}
                       for i in range(num_tasks)}

    for i in range(num_tasks):
        end_performance[i]['acc'] = mean_acc[i]
        end_performance[i]['auroc'] = mean_auroc[i]
        end_performance[i]['auprc'] = mean_auprc[i]
        end_performance[i]['std_acc'] = std_acc[i]
        end_performance[i]['std_auroc'] = std_auroc[i]
        end_performance[i]['std_auprc'] = std_auprc[i]

    metrics = ['acc', 'auroc', 'auprc']  # possibilities: 'acc', 'auroc', 'auprc'
    print('\n\nMetrics at the end of each task training:\n', end_performance)
    plot_multiple_histograms(end_performance, num_tasks, metrics,
                             '#runs: %d, average task results, %s method, early stopping=%s (%s)' %
                             (args.repeat, args.agent_name, str(do_early_stopping), stopping_criteria),
                             colors[:len(metrics)], 'Metric value', min_y)

    print('\nMean time per task:')
    print([sum(mean_time_per_task[:i + 1]) / 60 for i in range(len(mean_time_per_task))])

    print('\nEnd performance')
    [print(k, ':', v['acc'], '+/-', v['std_acc']) for k, v in end_performance.items()]
    print([v['acc'] for k, v in end_performance.items()])


