from options.test_options import TestOptions
from models import create_model
import numpy as np
import os
import torch
from utils import pidfile, util, imutil, pbar
from utils import util
from utils import imutil
from torch.utils.data import DataLoader
from data.unpaired_dataset import UnpairedMaskDataset
from sklearn import metrics
import matplotlib.pyplot as plt
from PIL import Image
from IPython import embed
import pandas as pd
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

torch.backends.cudnn.benchmark = True

def run_eval(opt, output_dir):
    model = create_model(opt)
    model.setup(opt)
    model.eval()

    fake_label = opt.fake_class_id
    real_label = 1 - fake_label

    # values to track
    paths = []
    prediction_voted = []
    prediction_avg_after_softmax = []
    prediction_avg_before_softmax = []
    prediction_raw = []
    prediction_mask = []
    labels = []

    for data_path, label in zip([opt.real_im_path, opt.fake_im_path],
                                [real_label, fake_label]):
        dset = UnpairedMaskDataset(opt, data_path, label, is_val=True)
        dl = DataLoader(dset, batch_size=opt.batch_size, shuffle=False,
                        num_workers=opt.nThreads, pin_memory=False)

        for i, data in enumerate(dl):
            # set model inputs
            if opt.model == 'patch_inconsistency_discriminator':
                ims = data['img'].to(opt.gpu_ids[0])
                mks = data['mask'].to(opt.gpu_ids[0])
                pred_labels = (torch.ones(ims.shape[0], dtype=torch.long)
                            * label).to(opt.gpu_ids[0])
                inputs = dict(ims=ims, masks=mks, labels=pred_labels)
            else:
                ims = data['img'].to(opt.gpu_ids[0])
                pred_labels = (torch.ones(ims.shape[0], dtype=torch.long)
                            * label).to(opt.gpu_ids[0])
                inputs = dict(ims=ims, labels=pred_labels)

            # forward pass
            model.reset()
            model.set_input(inputs)
            model.test(True)
            predictions = model.get_predictions()

            # update counts
            labels.append(pred_labels.cpu().numpy())
            prediction_voted.append(predictions.vote)
            prediction_avg_before_softmax.append(predictions.before_softmax)
            prediction_avg_after_softmax.append(predictions.after_softmax)
            prediction_raw.append(predictions.raw)
            prediction_mask.append(predictions.mask)
            paths.extend(data['path'])

    # compute and save metrics
    if opt.model.split('_')[0] == 'patch':
        # save precision, recall, AP metrics on voted predictions
        compute_metrics(np.concatenate(prediction_voted),
                        np.concatenate(labels),
                        os.path.join(output_dir, 'metrics_voted'))

        # save precision, recall, AP metrics, avg before softmax
        compute_metrics(np.concatenate(prediction_avg_before_softmax),
                        np.concatenate(labels),
                        os.path.join(output_dir, 'metrics_avg_before_softmax'))

        # save precision, recall, AP metrics, avg after softmax
        compute_metrics(np.concatenate(prediction_avg_after_softmax),
                        np.concatenate(labels),
                        os.path.join(output_dir, 'metrics_avg_after_softmax'))

        # save precision, recall, AP metrics, on raw patches
        # this can be slow, so will not plot AP curve
        patch_preds = np.concatenate(prediction_raw, axis=0) # N2HW
        patch_preds = patch_preds.transpose(0, 2, 3, 1) # NHW2
        n, h, w, c = patch_preds.shape
        patch_labels = np.concatenate(labels, axis=0)[:, None, None]
        patch_labels = np.tile(patch_labels, (1, h, w))
        patch_preds = patch_preds.reshape(-1, 2)
        patch_labels = patch_labels.reshape(-1)
        compute_metrics(patch_preds, patch_labels,
                        os.path.join(output_dir, 'metrics_patch'),
                        plot=False)

        if opt.visualize:
            if opt.model == 'patch_inconsistency_discriminator':
                pred_output = {
                    'vote': np.concatenate(prediction_voted),
                    'before_softmax': np.concatenate(prediction_avg_before_softmax),
                    'after_softmax': np.concatenate(prediction_avg_after_softmax),
                    'mask': np.concatenate(prediction_mask)
                }[opt.average_mode]
            else:
                pred_output = {
                    'vote': np.concatenate(prediction_voted),
                    'before_softmax': np.concatenate(prediction_avg_before_softmax),
                    'after_softmax': np.concatenate(prediction_avg_after_softmax)
                }[opt.average_mode]
            vis_dir = os.path.join(output_dir, 'vis')
            transform = dset.transform
            model.visualize(pred_output, paths, np.concatenate(labels),
                            transform, fake_label,
                            os.path.join(vis_dir, 'fakes'),
                            opt.topn)
            model.visualize(pred_output, paths, np.concatenate(labels),
                            transform, real_label,
                            os.path.join(vis_dir, 'reals'),
                            opt.topn)
    else:
        # save precision, recall, AP metrics, on non-patch-based model
        compute_metrics(np.concatenate(prediction_raw),
                        np.concatenate(labels),
                        os.path.join(output_dir, 'metrics'))


def run_eval_openmfc(opt):
    model = create_model(opt)
    model.setup(opt)
    model.eval()

    fake_label = opt.fake_class_id
    label = fake_label

    MFC18_ref_csv = opt.fake_im_path + \
        '/../../indexes/MFC18_Eval-manipulation-image-index.csv'
    df = pd.read_csv(MFC18_ref_csv, sep='|')
    df = df.drop('TaskID', axis=1)
    df = df.drop('ProbeFileName', axis=1)
    df = df.drop('ProbeWidth', axis=1)
    df = df.drop('ProbeHeight', axis=1)
    df = df.drop('ProbeFileSize', axis=1)
    df = df.drop('HPDeviceID', axis=1)
    df['ConfidenceScore'] = 0
    df['OutputProbeMaskFileName'] = np.nan
    df['ProbeStatus'] = 'NonProcessed'
    df['ProbeOptOutPixelValue'] = np.nan

    data_path = opt.fake_im_path
    dset = UnpairedMaskDataset(opt, data_path, label, is_val=True)
    dl = DataLoader(dset, batch_size=opt.batch_size, shuffle=False,
                    num_workers=opt.nThreads, pin_memory=False)

    for i, data in enumerate(dl):
        # set model inputs
        ims = data['img'].to(opt.gpu_ids[0])
        mks = data['mask'].to(opt.gpu_ids[0])
        id_lst = data['path']

        for j in range(len(id_lst)):
            tmp = id_lst[j]
            tmp = tmp.split('/')[-1]
            tmp = tmp.split('.')[0]
            id_lst[j] = tmp

        pred_labels = (torch.ones(ims.shape[0], dtype=torch.long)
                       * label).to(opt.gpu_ids[0])
        inputs = dict(ims=ims, masks=mks, labels=pred_labels)

        # forward pass
        model.reset()
        model.set_input(inputs)
        model.test(True)
        predictions = model.get_predictions()
        tmp_lst = np.asarray(predictions.after_softmax)
        tmp_lst = np.reshape(tmp_lst, (-1, 2))
        ConfSc = tmp_lst[:, 1]

        for i in range(len(id_lst)):
            df.loc[df['ProbeFileID'].str.match(
                id_lst[i]), 'ConfidenceScore'] = ConfSc[i]
            df.loc[df['ProbeFileID'].str.match(
                id_lst[i]), 'ProbeStatus'] = 'Processed'

    os.makedirs("openmfc", exist_ok=True)
    os.makedirs("openmfc/{}".format(opt.name), exist_ok=True)
    df.to_csv('openmfc/{}/test.csv'.format(opt.name), sep='|')



def compute_metrics(predictions, labels, save_path, threshold=0.5, plot=True):
    # save precision, recall, AP metrics on voted predictions
    print("Computing metrics for %s" % save_path)
    assert(len(np.unique(labels)) == 2)
    assert(np.ndim(predictions) == 2)
    assert(predictions.shape[1] == 2)
    assert(len(labels) == predictions.shape[0])
    # predictions should be Nx2 np array
    # labels should be (N,) array with 0,1 values
    ap = metrics.average_precision_score(labels, predictions[:, 1])
    precision, recall, thresholds = metrics.precision_recall_curve(
        labels, predictions[:, 1])
    acc = metrics.accuracy_score(labels, np.argmax(predictions, axis=1))
    auc = metrics.roc_auc_score(labels, np.argmax(predictions, axis=1))
    fpr, tpr, threshold = metrics.roc_curve(labels, np.argmax(predictions, axis=1))
    fnr = 1 - tpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    print("ap: %0.6f, acc: %0.6f, auc: %0.6f, eer: %0.6f" % (ap, acc, auc, eer_threshold))

    np.savez(save_path + '.npz', ap=ap, precision=precision,
             recall=recall, thresholds=thresholds, acc=acc, n=len(labels))
    if plot:
        f, ax = plt.subplots(1, 1)
        ax.plot(recall, precision)
        ax.set_xlabel('recall')
        ax.set_ylabel('precision')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        f.savefig(save_path + '.pdf')

if __name__ == '__main__':
    opt = TestOptions().parse()
    print("Evaluating model: %s epoch %s" % (opt.name, opt.which_epoch))
    print("On dataset (real): %s" % (opt.real_im_path))
    print("And dataset (fake): %s" % (opt.fake_im_path))
    expdir = opt.name
    dataset_name = opt.dataset_name
    output_dir = os.path.join(opt.results_dir, expdir, opt.partition,
                              'epoch_%s' % opt.which_epoch, dataset_name)
    print(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # check if checkpoint is out of date (e.g. if model is still training)
    redo = opt.force_redo
    ckpt_path = os.path.join(opt.checkpoints_dir, opt.name, '%s_net_D.pth'
                             % opt.which_epoch)
    timestamp_path = os.path.join(output_dir, 'timestamp_%s_net_D.txt'
                                  % opt.which_epoch)
    if util.check_timestamp(ckpt_path, timestamp_path):
        redo = True
        util.update_timestamp(ckpt_path, timestamp_path)
    pidfile.exit_if_job_done(output_dir, redo=redo)
    if opt.dataset_name == 'openmfc':
        run_eval_openmfc(opt)
    else:
        run_eval(opt, output_dir)
    pidfile.mark_job_done(output_dir)

