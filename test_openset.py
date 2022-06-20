import collections
import os
from lib.cmdparser import parser
import lib.Datasets.datasets as datasets
import lib.Models.architectures as architectures
from lib.Utility.visualization import *
from lib.OpenSet.meta_recognition import *
from lib.Utility.metrics import caculate_metrics
import numpy as np


def main():
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Command line options
    args = parser.parse_args()
    print("Command line options:")
    for arg in vars(args):
        print(arg, getattr(args, arg))

    # choose dataset evaluation function to import (e.g. variational will operate on z values)
    if args.train_var:
        from lib.Training.evaluate import eval_var_dataset as eval_dataset
        from lib.Training.evaluate import eval_var_openset_dataset as eval_openset_dataset
    else:
        from lib.Training.evaluate import eval_dataset as eval_dataset
        from lib.Training.evaluate import eval_openset_dataset as eval_openset_dataset

    # Get the dataset which has been trained and the corresponding number of classes
    data_init_method = getattr(datasets, args.dataset)
    dataset = data_init_method(torch.cuda.is_available(), args)
    num_classes = dataset.num_classes
    net_input, _ = next(iter(dataset.train_loader))
    num_colors = net_input.size(1)

    # Split a part of the non-used dataset to use as validation set for determining open set (e.g entropy)
    # rejection thresholds
    split_perc = 0.5
    split_sets = torch.utils.data.random_split(dataset.valset,
                                               [int((1 - split_perc) * len(dataset.valset)),
                                                int(split_perc * len(dataset.valset))])

    # overwrite old set and create new split set to determine thresholds/priors
    dataset.valset = split_sets[0]
    dataset.threshset = split_sets[1]

    # overwrite old data loader and create new loader for thresh set
    is_gpu = torch.cuda.is_available()
    dataset.val_loader = torch.utils.data.DataLoader(dataset.valset, batch_size=args.batch_size, shuffle=False,
                                                     num_workers=args.workers, pin_memory=is_gpu, sampler=None)
    dataset.threshset_loader = torch.utils.data.DataLoader(dataset.threshset, batch_size=args.batch_size, shuffle=False,
                                                           num_workers=args.workers, pin_memory=is_gpu, sampler=None)

    # Load open set datasets
    openset_datasets_names = args.openset_datasets.strip().split(',')
    openset_datasets = []
    for openset_dataset in openset_datasets_names:
        openset_data_init_method = getattr(datasets, 'OpenSetDataset')
        openset_datasets.append(openset_data_init_method(torch.cuda.is_available(), args, openset_dataset))

    # Initialize empty model
    net_init_method = getattr(architectures, args.architecture)
    model = net_init_method(device, num_classes, num_colors, args).to(device)
    model = torch.nn.DataParallel(model).to(device)
    print(model)

    # load model (using the resume functionality)
    assert(os.path.isfile(args.resume)), "=> no model checkpoint found at '{}'".format(args.resume)

    # Fill the random model with the parameters of the checkpoint
    print("=> loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)
    best_prec = checkpoint['best_prec']
    best_loss = checkpoint['best_loss']
    print("Saved model's validation accuracy: ", best_prec)
    print("Saved model's validation loss: ", best_loss)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    # set the save path to the directory from which the model has been loaded
    save_path = os.path.dirname(args.resume)

    # start of the model evaluation on the training dataset and fitting
    print("Evaluate the training dataset. This may take a while...")
    dataset_eval_dict_train = eval_dataset(model, dataset.train_loader, num_classes, device,
                                           latent_var_samples=args.var_samples, model_var_samples=args.model_samples)
    print("Evaluation accuracy: ", dataset_eval_dict_train["accuracy"])

    # Get the mean of z for correctly classified data inputs
    mean_zs = get_means(dataset_eval_dict_train["zs_correct"])

    # calculate each correctly classified example's distance to the mean z
    distances_to_z_means_correct_train = calc_distances_to_means(mean_zs, dataset_eval_dict_train["zs_correct"],
                                                                 args.distance_function)

    weibull_models, valid_weibull = fit_weibull_models(distances_to_z_means_correct_train, args.openset_weibull_tailsize)
    assert valid_weibull, "Weibull fit is not valid"

    # Fitting on train dataset complete. Determine rejection thresholds/priors on the created split set
    print("Determine rejection thresholds on the split set. This may take a while...")
    threshset_eval_dict = eval_dataset(model, dataset.threshset_loader, num_classes, device,
                                       latent_var_samples=args.var_samples, model_var_samples=args.model_samples)

    # Again calculate distances to mean z
    print("Accuracy of split set: ", threshset_eval_dict["accuracy"])
    distances_to_z_means_threshset = calc_distances_to_means(mean_zs, threshset_eval_dict["zs_correct"],
                                                             args.distance_function)
    # get Weibull outlier probabilities for thresh set
    outlier_probs_threshset = calc_outlier_probs(weibull_models, distances_to_z_means_threshset)
    threshset_classification = calc_openset_classification(outlier_probs_threshset, num_classes,
                                                           num_outlier_threshs=100)
    # also check outlier detection based on entropy
    max_entropy = np.max(threshset_eval_dict["out_entropy"])
    threshset_entropy_classification = calc_entropy_classification(threshset_eval_dict["out_entropy"],
                                                                   max_entropy,
                                                                   num_outlier_threshs=100)

    # determine rejection priors based on 5% of the split data considered as inlying
    if (np.array(threshset_classification["outlier_percentage"]) <= 0.01).any():
        EVT_prior_index = np.argwhere(np.array(threshset_classification["outlier_percentage"])
                                      <= 0.01)[0][0]
        EVT_prior = threshset_classification["thresholds"][EVT_prior_index]
    else:
        EVT_prior = 0.5
        EVT_prior_index = 50

    if (np.array(threshset_entropy_classification["entropy_outlier_percentage"]) <= 0.01).any():
        entropy_threshold_index = np.argwhere(np.array(threshset_entropy_classification["entropy_outlier_percentage"])
                                              <= 0.01)[0][0]
        entropy_threshold = threshset_entropy_classification["entropy_thresholds"][entropy_threshold_index]
    else:
        entropy_threshold = np.median(threshset_entropy_classification["entropy_thresholds"])
        entropy_threshold_index = 50

    print("EVT prior: " + str(EVT_prior) + "; Entropy threshold: " + str(entropy_threshold))

    # Beginning of all testing/open set recognition on test and unknown sets.
    # We evaluate the validation set to later evaluate trained dataset's statistical inlier/outlier estimates.
    print("Evaluate the validation set. This may take a while...")
    dataset_eval_dict = eval_dataset(model, dataset.val_loader, num_classes, device,
                                     latent_var_samples=args.var_samples, model_var_samples=args.model_samples)

    # Again calculate distances to mean z
    print("Accuracy of validation set: ", dataset_eval_dict["accuracy"])
    distances_to_z_means_correct = calc_distances_to_means(mean_zs, dataset_eval_dict["zs_correct"],
                                                           args.distance_function)

    # Evaluate outlier probability of trained dataset's validation set
    outlier_probs_correct = calc_outlier_probs(weibull_models, distances_to_z_means_correct)

    dataset_classification_correct = calc_openset_classification(outlier_probs_correct, num_classes,
                                                                 num_outlier_threshs=100)
    dataset_entropy_classification_correct = calc_entropy_classification(dataset_eval_dict["out_entropy"],
                                                                         max_entropy,
                                                                         num_outlier_threshs=100)

    print('EVT outlier percentage: ' + str(dataset_classification_correct["outlier_percentage"][EVT_prior_index]))
    print('entropy outlier percentage: ' + str(dataset_entropy_classification_correct["entropy_outlier_percentage"][entropy_threshold_index]))

    # Repeat process for open set recognition on unseen datasets
    openset_dataset_eval_dicts = collections.OrderedDict()
    openset_outlier_probs_dict = collections.OrderedDict()
    openset_classification_dict = collections.OrderedDict()
    openset_entropy_classification_dict = collections.OrderedDict()

    X = []
    label = []
    for od, openset_dataset in enumerate(openset_datasets):
        print("Evaluate the open set dataset: " + openset_datasets_names[od] + ". This may take a while...")
        openset_dataset_eval_dict = eval_openset_dataset(model, openset_dataset.val_loader, num_classes,
                                                         device, latent_var_samples=args.var_samples,
                                                         model_var_samples=args.model_samples)

        openset_distances_to_z_means = calc_distances_to_means(mean_zs, openset_dataset_eval_dict["zs"],
                                                               args.distance_function)

        openset_outlier_probs = calc_outlier_probs(weibull_models, openset_distances_to_z_means)

        # getting outlier classification accuracies across the entire datasets
        openset_classification = calc_open_evt_classification(openset_outlier_probs, num_classes,
                                                             num_outlier_threshs=100)

        openset_entropy_classification = calc_open_entropy_classification(openset_dataset_eval_dict["out_entropy"],
                                                                     max_entropy, num_outlier_threshs=100)

        # dictionary of dictionaries: per datasetname one dictionary with respective values
        openset_dataset_eval_dicts[openset_datasets_names[od]] = openset_dataset_eval_dict
        openset_outlier_probs_dict[openset_datasets_names[od]] = openset_outlier_probs
        openset_classification_dict[openset_datasets_names[od]] = openset_classification
        openset_entropy_classification_dict[openset_datasets_names[od]] = openset_entropy_classification

        import itertools
        new_zs = list(itertools.chain(*openset_dataset_eval_dict["zs"]))
        for i in range(150):
            X.append(new_zs[i].cpu().numpy())
            label.append(od)

    # print outlier rejection values for all unseen unknown datasets
    known_total = 0.0
    unknown_total = 0.0
    evt_known_correct = 0
    evt_unknown_correct = 0
    entropy_known_correct = 0
    entropy_unknown_correct = 0
    # CWRU
    dataset_classes = ['Normal', 'IR007', 'IR014', 'IR021', 'OR007', 'OR014', 'OR021', 'B007', 'B014', 'B021']

    known_classes = [dataset_classes[int(x)] for x in args.known_classes]
    for other_data_name, other_data_dict in openset_classification_dict.items():
        print(other_data_name + ' EVT outliers: ' +
              str(other_data_dict["outlier_num"][EVT_prior_index]) + '/' +
              str(other_data_dict["total"]))
        if other_data_name in known_classes:
            evt_known_correct += other_data_dict["total"] - other_data_dict["outlier_num"][EVT_prior_index]
            known_total += other_data_dict["total"]
        else:
            evt_unknown_correct += other_data_dict["outlier_num"][EVT_prior_index]
            unknown_total += other_data_dict["total"]

    for other_data_name, other_data_dict in openset_entropy_classification_dict.items():
        print(other_data_name + ' entropy outliers: ' +
              str(other_data_dict["outlier_num"][entropy_threshold_index]) + '/' +
              str(other_data_dict["total"]))
        if other_data_name in known_classes:
            entropy_known_correct += other_data_dict["total"] - other_data_dict["outlier_num"][entropy_threshold_index]
        else:
            entropy_unknown_correct += other_data_dict["outlier_num"][entropy_threshold_index]

    # EVT
    evt_tp = evt_unknown_correct
    evt_tn = evt_known_correct
    evt_fp = known_total - evt_known_correct
    evt_fn = unknown_total - evt_unknown_correct
    print('Result (EVT)——tp: %d, tn: %d, fp: %d, fn: %d' % (evt_tp, evt_tn, evt_fp, evt_fn))
    evt_metrics = caculate_metrics(evt_tp, evt_tn, evt_fp, evt_fn)
    # entropy
    entropy_tp = entropy_unknown_correct
    entropy_tn = entropy_known_correct
    entropy_fp = known_total - entropy_known_correct
    entropy_fn = unknown_total - entropy_unknown_correct
    print('Result (entropy)——tp: %d, tn: %d, fp: %d, fn: %d' % (entropy_tp, entropy_tn, entropy_fp, entropy_fn))
    entropy_metrics = caculate_metrics(entropy_tp, entropy_tn, entropy_fp, entropy_fn)

    print('EVT——Accuracy：%f(%d/%d)' % (evt_metrics["total_accuracy"], evt_known_correct + evt_unknown_correct, known_total + unknown_total))
    print('EVT——Accuracy of known：%f(%d/%d)' % (evt_metrics["known_accuracy"], evt_known_correct, known_total))
    print('EVT——Precision of unknown: %f. Recall of unknown: %f, f1-score: %f)' % (evt_metrics["precision"], evt_metrics["recall"], evt_metrics["f1"]))

    print('entropy——Accuracy：%f(%d/%d)' % (entropy_metrics["total_accuracy"], entropy_known_correct + entropy_unknown_correct, known_total + unknown_total))
    print('entropy——Accuracy of known：%f(%d/%d)' % (entropy_metrics["known_accuracy"], entropy_known_correct, known_total))
    print('entropy——Precision of unknown: %f. Recall of unknown: %f, f1-score: %f)' % (entropy_metrics["precision"], entropy_metrics["recall"], entropy_metrics["f1"]))

    # T-SNE 2d visualize
    tsne(X, label)


if __name__ == '__main__':
    main()
