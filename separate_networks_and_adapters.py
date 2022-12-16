import time
import argparse
from models import *
from superposition import *
from prepare_data import *
from Split_CIFAR_100_preparation import get_dataset
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='Upper_bound', choices=['Upper_bound', 'Lower_bound', 'Adapters'])
    parser.add_argument('--network_type', type=str, default='transformer', choices=['transformer', 'MLP', 'CNN'])
    parser.add_argument('--num_runs', type=int, default=5)
    args, _ = parser.parse_known_args()

    adapter_trainable_layers = ["transformer_encoder.layers.0.layer_norm_trainable.weight",
                                "transformer_encoder.layers.0.layer_norm_trainable.bias",
                                "transformer_encoder.layers.0.adapter_layer.bottleneck_mlp.0.weight",
                                "transformer_encoder.layers.0.adapter_layer.bottleneck_mlp.0.bias",
                                "transformer_encoder.layers.0.adapter_layer.bottleneck_mlp.2.weight",
                                "transformer_encoder.layers.0.adapter_layer.bottleneck_mlp.2.bias",
                                "transformer_encoder.layers.0.adapter_layer.one2one.weight",
                                "mlp.2.weight", "mlp.2.bias"]

    first_average = 'average'     # show 'average' results until current task
    use_MLP = False
    use_mask = False

    # method options: 'adapters', 'upper bound' (use separate transformer networks for each task), 'lower bound' (use single transformer network for all tasks)
    if args.method == 'Upper_bound':
        method = 'upper bound'
    elif args.method == 'Lower_bound':
        method = 'lower bound'
    elif args.method == 'Adapters':
        method = 'adapters'

    input_size = 32 if args.network_type == 'transformer' else (3, 32, 32)
    num_heads = 4
    num_layers = 1      # number of transformer encoder layers
    dim_feedforward = 1024
    num_classes = 2 if args.network_type == 'transformer' else 10
    bottleneck_size = 4    # only used with adapters
    standardize_input = False
    restore_best_auroc = False
    do_early_stopping = True
    stopping_criteria = 'auroc' if args.network_type == 'transformer' else 'acc'   # possibilities: 'acc', 'auroc', 'auprc'

    batch_size = 128
    num_runs = args.num_runs
    num_tasks = 6 if args.network_type == 'transformer' else 10
    num_epochs = 500 if method == 'adapters' else 50
    learning_rate = 0.0001 if args.network_type == 'CNN' else 0.001

    task_names = [['HS', 'SA', 'S', 'SA_2', 'C', 'HD'],
                  ['C', 'HD', 'SA', 'HS', 'SA_2', 'S'],
                  ['SA', 'S', 'HS', 'SA_2', 'HD', 'C'],
                  ['HD', 'SA_2', 'SA', 'C', 'S', 'HS'],
                  ['SA', 'HS', 'C', 'SA_2', 'HD', 'S']]

    # load Split CIFAR-100
    if args.network_type == 'MLP':
        split_cifar_100 = get_dataset('nn', (32, 32, 3))
    elif args.network_type == 'CNN':
        split_cifar_100 = get_dataset('cnn', (32, 32, 3))

    # Train model for 'num_runs' runs for 'num_tasks' tasks
    acc_arr = np.zeros((num_runs, num_tasks))
    auroc_arr = np.zeros((num_runs, num_tasks))
    auprc_arr = np.zeros((num_runs, num_tasks))

    acc_epoch = np.zeros((num_runs, num_tasks * num_epochs))
    auroc_epoch = np.zeros((num_runs, num_tasks * num_epochs))
    auprc_epoch = np.zeros((num_runs, num_tasks * num_epochs))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    task_epochs_all = []
    times_per_run = []
    times_per_task = np.zeros((num_runs, num_tasks))

    for r in range(num_runs):
        print('- - Run %d - -' % (r + 1))

        start_time = time.time()
        previous_time = start_time

        all_tasks_test_data = []
        task_epochs = []

        if method == 'adapters':
            model = AdapterTransformer(input_size, num_heads, num_layers, dim_feedforward, num_classes, bottleneck_size).to(device)

            for name, params in model.named_parameters():
                if name not in adapter_trainable_layers:
                    params.requires_grad = False
                print('Requires_grad: %s,  layer name: "%s", number of layer parameters: %d' % (str(params.requires_grad), name, params.numel()))
        elif method == 'lower bound':
            if args.network_type == 'MLP':
                use_MLP = True
                model = MLP(input_size, num_classes, data='CV').to(device)  # or data='NLP'
            elif args.network_type == 'CNN':
                use_MLP = True  # even though the model is CNN, this variable denotes we do not use transformers
                model = CNN(input_size, num_classes).to(device)
            elif args.network_type == 'transformer':
                model = MyTransformer(input_size, num_heads, num_layers, dim_feedforward, num_classes).to(device)

        for t in range(num_tasks):
            print('- Task %d -' % (t + 1))

            if method == 'upper bound':
                if args.network_type == 'MLP':
                    use_MLP = True
                    model = MLP(input_size, num_classes, data='CV').to(device)  # or data='NLP'
                elif args.network_type == 'CNN':
                    use_MLP = True  # even though the model is CNN, this variable denotes we do not use transformers
                    model = CNN(input_size, num_classes).to(device)
                elif args.network_type == 'transformer':
                    model = MyTransformer(input_size, num_heads, num_layers, dim_feedforward, num_classes).to(device)

                if t == 0:
                    print(model)

            criterion = torch.nn.CrossEntropyLoss().cuda()

            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2,
                                                                   threshold=0.0001, min_lr=1e-8, verbose=True)

            # stop training if none of the validation metrics improved from the previous epoch (accuracy, AUROC, AUPRC)
            if do_early_stopping:
                early_stopping = (0, 0, 0)  # (accuracy, AUROC, AUPRC)

            print('Number of trainable parameters: ', count_trainable_parameters(model))
            # print(model)
            # summary(model, [(batch_size, 256, 32), (batch_size, 256)])

            best_auroc_val = 0

            # prepare data
            if args.network_type == 'transformer':
                X, y, mask = get_data(task_names[r][t])

                if standardize_input:
                    for i in range(X.shape[0]):
                        X[i, :, :] = torch.from_numpy(StandardScaler().fit_transform(X[i, :, :]))

                        # where samples are padded, make zeros again
                        mask_i = torch.ones(X.shape[1]) - mask[i, :]
                        for j in range(X.shape[2]):
                            X[i, :, j] = X[i, :, j] * mask_i

                # split data into train, validation and test set
                y = torch.max(y, 1)[1]  # change one-hot-encoded vectors to numbers
                permutation = torch.randperm(X.size()[0])
                X = X[permutation]
                y = y[permutation]
                mask = mask[permutation]
                index_val = round(0.8 * len(permutation))
                index_test = round(0.9 * len(permutation))

                X_train, y_train, mask_train = X[:index_val, :, :], y[:index_val], mask[:index_val, :]
                X_val, y_val, mask_val = X[index_val:index_test, :, :], y[index_val:index_test], mask[index_val:index_test, :]
                X_test, y_test, mask_test = X[index_test:, :, :], y[index_test:], mask[index_test:, :]
            elif args.network_type == 'MLP' or args.network_type == 'CNN':
                X_train, y_train, X_test, y_test = split_cifar_100[t]
                y_train = torch.max(y_train, 1)[1]  # change one-hot-encoded vectors to numbers
                y_test = torch.max(y_test, 1)[1]  # change one-hot-encoded vectors to numbers
                permutation = torch.randperm(X_train.size()[0])
                X_train = X_train[permutation]
                y_train = y_train[permutation]

                mask_train_val = torch.FloatTensor(
                    [0] * len(X_train))  # only for the purpose of compatibility with transformer network
                mask_test = torch.FloatTensor(
                    [0] * len(X_test))  # only for the purpose of compatibility with transformer network
                index_val = round(0.9 * len(permutation))  # use 10% of test set as validation set

                if args.network_type == 'MLP':
                    X_val, y_val, mask_val = X_train[index_val:, :], y_train[index_val:], mask_train_val[index_val:]
                    X_train, y_train, mask_train = X_train[:index_val, :], y_train[:index_val], mask_train_val[:index_val]

                elif args.network_type == 'CNN':
                    X_val, y_val, mask_val = X_train[index_val:, :, :, :], y_train[index_val:], mask_train_val[index_val:]
                    X_train, y_train, mask_train = X_train[:index_val, :, :, :], y_train[:index_val], mask_train_val[:index_val]

            train_dataset = TensorDataset(X_train, y_train, mask_train)
            val_dataset = TensorDataset(X_val, y_val, mask_val)
            test_dataset = TensorDataset(X_test, y_test, mask_test)

            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

            all_tasks_test_data.append(test_loader)

            for epoch in range(num_epochs):
                model.train()
                model = model.cuda()

                for batch_X, batch_y, batch_mask in train_loader:
                    if torch.cuda.is_available():
                        batch_X = batch_X.cuda()
                        batch_y = batch_y.cuda()
                        batch_mask = batch_mask.cuda()

                        if not use_mask:
                            batch_mask = None

                    if use_MLP:
                        outputs = model.forward(batch_X)
                    else:
                        outputs = model.forward(batch_X, batch_mask)

                    optimizer.zero_grad()
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

                # check validation set
                model.eval()
                with torch.no_grad():
                    val_outputs = []

                    for batch_X, batch_y, batch_mask in val_loader:
                        if torch.cuda.is_available():
                            batch_X = batch_X.cuda()
                            batch_mask = batch_mask.cuda()

                            if not use_mask:
                                batch_mask = None

                        if use_MLP:
                            outputs = model.forward(batch_X)
                        else:
                            outputs = model.forward(batch_X, batch_mask)

                        val_outputs.append(outputs)

                    val_acc, val_auroc, val_auprc = get_stats(val_outputs, y_val)
                    val_loss = criterion(torch.cat(val_outputs, dim=0), y_val.cuda())

                    print("Epoch: %d --- val acc: %.2f, val AUROC: %.2f, val AUPRC: %.2f, val loss: %.3f" %
                          (epoch, val_acc * 100, val_auroc * 100, val_auprc * 100, val_loss))

                    scheduler.step(val_auroc)
                    if restore_best_auroc and val_auroc > best_auroc_val:
                        best_auroc_val = val_auroc
                        torch.save(model.state_dict(), 'models/model_best.pt')

                if do_early_stopping:
                    # check early stopping criteria
                    # if val_acc > early_stopping[0] or val_auroc > early_stopping[1] or val_auprc > early_stopping[2]:     # improvement on acc, auroc, auprc
                    if stopping_criteria == 'acc' and val_acc > early_stopping[0]:   # improvement only on acc
                        early_stopping = (val_acc, val_auroc, val_auprc)
                    elif stopping_criteria == 'auroc' and val_auroc > early_stopping[1]:   # improvement only on auroc
                        early_stopping = (val_acc, val_auroc, val_auprc)
                    elif stopping_criteria == 'auprc' and val_auprc > early_stopping[2]:   # improvement only on auprc
                        early_stopping = (val_acc, val_auroc, val_auprc)
                    else:   # stop training
                        print('Early stopped - %s got worse in this epoch.' % stopping_criteria)
                        task_epochs.append(epoch)

                        acc_e, auroc_e, auprc_e = evaluate_results(model, None, None, all_tasks_test_data,
                                                                   False, t, first_average, use_MLP, batch_size)

                        acc_epoch[r, (t * num_epochs) + epoch] = acc_e
                        auroc_epoch[r, (t * num_epochs) + epoch] = auroc_e
                        auprc_epoch[r, (t * num_epochs) + epoch] = auprc_e
                        break

                # track results with or without superposition
                if epoch == num_epochs - 1:   # calculate results for each epoch or only the last epoch in task
                    task_epochs.append(epoch)
                    acc_e, auroc_e, auprc_e = evaluate_results(model, None, None, all_tasks_test_data,
                                                                False, t, first_average, use_MLP, batch_size)
                else:
                    acc_e, auroc_e, auprc_e = 0, 0, 0

                acc_epoch[r, (t * num_epochs) + epoch] = acc_e
                auroc_epoch[r, (t * num_epochs) + epoch] = auroc_e
                auprc_epoch[r, (t * num_epochs) + epoch] = auprc_e

            # check test set
            if restore_best_auroc:
                model.load_state_dict(torch.load('models/model_best.pt'))
            model.eval()
            with torch.no_grad():
                test_outputs = []

                for batch_X, batch_y, batch_mask in test_loader:
                    if torch.cuda.is_available():
                        batch_X = batch_X.cuda()
                        batch_mask = batch_mask.cuda()

                        if not use_mask:
                            batch_mask = None

                    if use_MLP:
                        outputs = model.forward(batch_X)
                    else:
                        outputs = model.forward(batch_X, batch_mask)

                    test_outputs.append(outputs)

                test_acc, test_auroc, test_auprc = get_stats(test_outputs, y_test)

                print("TEST: test acc: %.2f, test AUROC: %.2f, test AUPRC: %.2f" %
                      (test_acc * 100, test_auroc * 100, test_auprc * 100))

                predicted = np.argmax(torch.cat(test_outputs, dim=0).cpu().detach().numpy(), axis=1).ravel()
                # print('Classification report:', classification_report(y_test.cpu().detach().numpy(), predicted))
                print('Confusion matrix:\n', confusion_matrix(y_test.cpu().detach().numpy(), predicted, labels=list(range(num_classes))))

            # store statistics
            acc_arr[r, t] = test_acc * 100
            auroc_arr[r, t] = test_auroc * 100
            auprc_arr[r, t] = test_auprc * 100

            print('Elapsed time so far: %.2f s, %.2f min' % (time.time() - start_time, (time.time() - start_time) / 60))
            times_per_task[r, t] = time.time() - previous_time  # saved in seconds
            previous_time = time.time()

        task_epochs_all.append(task_epochs)

        end_time = time.time()
        time_elapsed = end_time - start_time
        times_per_run.append(time_elapsed)
        print('Time elapsed for this run:', round(time_elapsed, 2), 's')

    epochs_per_run = np.array(task_epochs_all) + 1    # +1 to be consistent with CL benchmarks
    print('\nEpochs per run: ', epochs_per_run)
    print('Times per run: ', times_per_run)
    print('Runs: %d,  Average time per run: %.2f +/- %.2f s, %.1f +/- %.1f min' %
          (num_runs, np.mean(np.array(times_per_run)), np.std(np.array(times_per_run)), np.mean(np.array(times_per_run)) / 60, np.std(np.array(times_per_run)) / 60))
    print('Runs: %d,  Average #epochs for all tasks: %.2f +/- %.2f\n' %
          (num_runs, np.mean(np.array([sum(l) for l in epochs_per_run])), np.std(np.array([sum(l) for l in epochs_per_run]))))

    # display mean and standard deviation per task
    mean_acc, std_acc = np.mean(acc_arr, axis=0), np.std(acc_arr, axis=0)
    mean_auroc, std_auroc = np.mean(auroc_arr, axis=0), np.std(auroc_arr, axis=0)
    mean_auprc, std_auprc = np.mean(auprc_arr, axis=0), np.std(auprc_arr, axis=0)

    mean_time_per_task, std_time_per_task = np.mean(times_per_task, axis=0), np.std(times_per_task, axis=0)

    print('\nMeans for each task separately:')
    for t in range(num_tasks):
        print('------------------------------------------')
        print('Mean time for task %d: %.2f +/- %.2f s,  %.2f +/- %.2f min' % (t + 1, mean_time_per_task[t], std_time_per_task[t], mean_time_per_task[t] / 60, std_time_per_task[t] / 60))
        print('Mean time until task %d: %.2f s,  %.2f min' % (t + 1, sum(mean_time_per_task[:t + 1]), sum(mean_time_per_task[:t + 1]) / 60))
        print('Task %d - Accuracy = %.1f +/- %.1f' % (t + 1, mean_acc[t], std_acc[t]))
        print('Task %d - AUROC    = %.1f +/- %.1f' % (t + 1, mean_auroc[t], std_auroc[t]))
        print('Task %d - AUPRC    = %.1f +/- %.1f' % (t + 1, mean_auprc[t], std_auprc[t]))

    '''
    for t in range(num_tasks):
        if t == 0:
            # s = 'Hate speech'
            s = permutations[permutation_index][0]
        elif t == 1:
            # s = 'IMDB sentiment analysis'
            s = permutations[permutation_index][1]
        elif t == 2:
            # s = 'SMS spam'
            s = permutations[permutation_index][2]
        elif t == 3:
            s = 'Amazon, Yelp sentiment analysis'
        elif t == 4:
            s = 'Clickbait'
        elif t == 5:
            s = 'Humor detection'

        if s == 'HS':
            s = 'Hate speech'
        elif s == 'SA':
            s = 'IMDB sentiment analysis'
        elif s == 'S':
            s = 'SMS spam'

        print('------------------------------------------')
        print('%s - Accuracy = %.1f +/- %.1f' % (s, mean_acc[t], std_acc[t]))
        print('%s - AUROC    = %.1f +/- %.1f' % (s, mean_auroc[t], std_auroc[t]))
        print('%s - AUPRC    = %.1f +/- %.1f' % (s, mean_auprc[t], std_auprc[t]))
    '''

    show_only_accuracy = False
    min_y = 50 if args.network_type == 'transformer' else 0
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    if do_early_stopping:
        vertical_lines_x = []
        for task_epochs in task_epochs_all:
            vertical_lines_x.append([sum(task_epochs[:i+1]) - 1 for i in range(len(task_epochs))])
        if num_runs == 1:
            vertical_lines_x = vertical_lines_x[0]
    else:
        vertical_lines_x = [((i + 1) * num_epochs) - 1 for i in range(num_tasks)]

    if method == 'lower bound':
        acc_epoch_no0 = remove_empty_values(acc_epoch, num_tasks, num_epochs)
        auroc_epoch_no0 = remove_empty_values(auroc_epoch, num_tasks, num_epochs)
        auprc_epoch_no0 = remove_empty_values(auprc_epoch, num_tasks, num_epochs)

        # display mean and standard deviation per epoch
        mean_acc, std_acc = np.mean(acc_epoch_no0, axis=0), np.std(acc_epoch_no0, axis=0)
        mean_auroc, std_auroc = np.mean(auroc_epoch_no0, axis=0), np.std(auroc_epoch_no0, axis=0)
        mean_auprc, std_auprc = np.mean(auprc_epoch_no0, axis=0), np.std(auprc_epoch_no0, axis=0)
    else:
        acc_arr_average = [[sum(row[:i+1]) / (i+1) for i in range(len(row))] for row in acc_arr]
        auroc_arr_average = [[sum(row[:i+1]) / (i+1) for i in range(len(row))] for row in auroc_arr]
        auprc_arr_average = [[sum(row[:i+1]) / (i+1) for i in range(len(row))] for row in auprc_arr]
        mean_acc, std_acc = np.mean(acc_arr_average, axis=0), np.std(acc_arr_average, axis=0)
        mean_auroc, std_auroc = np.mean(auroc_arr_average, axis=0), np.std(auroc_arr_average, axis=0)
        mean_auprc, std_auprc = np.mean(auprc_arr_average, axis=0), np.std(auprc_arr_average, axis=0)

    # if (not do_early_stopping) or (do_early_stopping and num_runs == 1):
    #     if show_only_accuracy:
    #         plot_multiple_results(num_tasks, num_epochs, first_average,
    #                               [mean_acc], [std_acc], ['Accuracy'],
    #                               '#runs: %d, %s task results, %s model' % (num_runs, first_average,
    #                               'Adapter' if method == 'adapters' else 'Separate networks'), colors[0],
    #                               'Epoch', 'Accuracy (%)', vertical_lines_x[:-1], min_y, 100)
    #     else:   # show all three metrics
    #         plot_multiple_results(num_tasks, num_epochs, first_average,
    #                               [mean_acc, mean_auroc, mean_auprc], [std_acc, std_auroc, std_auprc], ['Accuracy', 'AUROC', 'AUPRC'],
    #                               '#runs: %d, %s task results, %s model' % (num_runs, first_average,
    #                               'Adapter' if  method == 'adapters' else 'Separate networks'), colors,
    #                               'Epoch', 'Metric value', vertical_lines_x[:-1], min_y, 100)

    # save only values at the end of task learning (at vertical lines), both mean and std
    end_performance = {i: {'acc': 0, 'auroc': 0, 'auprc': 0, 'std_acc': 0, 'std_auroc': 0, 'std_auprc': 0}
                       for i in range(num_tasks)}

    for i in range(num_tasks):
        index = i
        end_performance[i]['acc'] = mean_acc[index]
        end_performance[i]['auroc'] = mean_auroc[index]
        end_performance[i]['auprc'] = mean_auprc[index]
        end_performance[i]['std_acc'] = std_acc[index]
        end_performance[i]['std_auroc'] = std_auroc[index]
        end_performance[i]['std_auprc'] = std_auprc[index]

    metrics = ['acc', 'auroc', 'auprc']    # possibilities: 'acc', 'auroc', 'auprc'
    print('\n\nMetrics at the end of each task training:\n', end_performance)
    plot_multiple_histograms(end_performance, num_tasks, metrics,
                             '#runs: %d, %s task results, %s model, %s' % (num_runs, first_average, method,
                             'ES on %s' % stopping_criteria if do_early_stopping else 'no ES'),
                             colors[:len(metrics)], 'Metric value', min_y)

    print('\nMean time per task:')
    print([sum(mean_time_per_task[:i + 1]) / 60 for i in range(len(mean_time_per_task))])

    print('\nEnd performance')
    [print(k, ':', v['acc'], '+/-', v['std_acc']) for k, v in end_performance.items()]
    print([v['acc'] for k, v in end_performance.items()])


