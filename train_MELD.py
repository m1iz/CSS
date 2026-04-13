import os

from torch.optim.lr_scheduler import ReduceLROnPlateau

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np, argparse, time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataloader import IEMOCAPDataset, MELDDataset
from model import MaskedNLLLoss, MaskedKLDivLoss, Transformer_Based_Model, PGM
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
import pickle as pk
import datetime
torch.backends.cudnn.enabled = False
#
def set_seed(seed=10068):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

def get_train_valid_sampler(trainset, valid=0.1, dataset='MELD'):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid * size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_MELD_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = MELDDataset('data/meld_multimodal_features.pkl')
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid, 'MELD')
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = MELDDataset('data/meld_multimodal_features.pkl', train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader


def get_IEMOCAP_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = IEMOCAPDataset()
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = IEMOCAPDataset(train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader


def train_or_eval_model(model, loss_function, kl_loss, dataloader, epoch, optimizer=None, train=False):
    losses, preds, labels, masks = [], [], [], []
    PGM_loss = PGM(loss_function, kl_loss)

    assert not train or optimizer != None
    if train:
        model.train()
    else:
        model.eval()

    for data in dataloader:
        if train:
            optimizer.zero_grad()

        textf, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        qmask = qmask.permute(1, 0, 2)
        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]
        outputs = model(textf, visuf, acouf, umask, qmask, lengths)
        labels_ = label.view(-1)

        shared_params = [

            *model.shared_transformer.parameters(),

            *model.textf_input.parameters(),
            *model.acouf_input.parameters(),
            *model.visuf_input.parameters(),

            *model.text_to_audio.parameters(),
            *model.text_to_vision.parameters(),
            *model.bias_gate_audio.parameters(),
            *model.bias_gate_vision.parameters(),

            *model.t_t.parameters(),
            *model.a_a.parameters(),
            *model.v_v.parameters(),

            *model.t_t_gate.parameters(),
            *model.t_gate.parameters(),
            *model.a_a_gate.parameters(),
            *model.a_gate.parameters(),
            *model.v_v_gate.parameters(),
            *model.v_gate.parameters(),

            *model.features_reduce_t.parameters(),
            *model.features_reduce_a.parameters(),
            *model.features_reduce_v.parameters(),

        ]

        loss, weights, individual_losses = PGM_loss(
            outputs,
            labels_,
            umask,
            shared_params
        )

        all_prob = outputs[4]
        pred_ = torch.argmax(all_prob.view(-1, all_prob.size()[2]), 1)

        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())
        losses.append(loss.item() * masks[-1].sum())

        if train:
            loss.backward()
            if args.tensorboard:
                for param in model.named_parameters():
                    if param[1].grad is not None:
                        writer.add_histogram(param[0], param[1].grad, epoch)

                writer.add_scalar('weights/task1_weight', weights[0], epoch)
                writer.add_scalar('weights/task2_weight', weights[1], epoch)
                writer.add_scalar('weights/task3_weight', weights[2], epoch)

                for i, task_loss in enumerate(losses):
                    writer.add_scalar(f'losses/task{i + 1}_loss', task_loss.item(), epoch)
                writer.add_scalar('losses/total_loss', loss.item(), epoch)

            optimizer.step()


    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan')

    avg_loss = round(np.sum(losses) / np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels, preds, sample_weight=masks) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='weighted') * 100, 2)

    # plot_tsne(outputs[9], labels_)
    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--lr', type=float, default=0.000005, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=32, metavar='BS', help='batch size')
    parser.add_argument('--hidden_dim', type=int, default=1024, metavar='hidden_dim', help='output hidden size')
    parser.add_argument('--n_head', type=int, default=8, metavar='n_head', help='number of heads')
    parser.add_argument('--epochs', type=int, default=100, metavar='E', help='number of epochs')
    parser.add_argument('--temp', type=int, default=4, metavar='temp', help='temp')
    parser.add_argument('--rank', type=int, default=16, metavar='temp', help='projection rank')
    parser.add_argument('--order', type=int, default=4, metavar='temp', help='fusion order')
    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')
    parser.add_argument('--class-weight', action='store_true', default=True, help='use class weights')
    parser.add_argument('--Dataset', default='MELD', help='dataset to train and test')
    parser.add_argument('--seed', type=int, default=10068, help='random seed')

    args = parser.parse_args()
    set_seed(args.seed)
    today = datetime.datetime.now()
    print(args)

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    if args.tensorboard:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter()

    cuda = args.cuda
    n_epochs = args.epochs
    batch_size = args.batch_size
    feat2dim = {'IS10': 1582, 'denseface': 342, 'MELD_audio': 300}
    D_audio = feat2dim['IS10'] if args.Dataset == 'IEMOCAP' else feat2dim['MELD_audio']
    D_visual = feat2dim['denseface']
    D_text = 1024

    D_m = D_audio + D_visual + D_text

    n_speakers = 9 if args.Dataset == 'MELD' else 2
    n_classes = 7 if args.Dataset == 'MELD' else 6 if args.Dataset == 'IEMOCAP' else 1

    print('temp {}'.format(args.temp))

    model = Transformer_Based_Model(args.Dataset, args.temp, D_text, D_visual, D_audio, args.n_head,args.rank,args.order,
                                    n_classes=n_classes,
                                    hidden_dim=args.hidden_dim,
                                    n_speakers=n_speakers,
                                    dropout=args.dropout)
    total_params = sum(p.numel() for p in model.parameters())
    print('total parameters: {}'.format(total_params))
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('training parameters: {}'.format(total_trainable_params))

    if cuda:
        model.cuda()

    kl_loss = MaskedKLDivLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)



    if args.Dataset == 'MELD':

        loss_function = MaskedNLLLoss()

        train_loader, valid_loader, test_loader = get_MELD_loaders(valid=0.0,
                                                                   batch_size=batch_size,
                                                                   num_workers=0)
    elif args.Dataset == 'IEMOCAP':
        loss_weights = torch.FloatTensor([1 / 0.086747,
                                          1 / 0.144406,
                                          1 / 0.227883,
                                          1 / 0.160585,
                                          1 / 0.127711,
                                          1 / 0.252668])

        loss_function = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)

        train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(valid=0.0,
                                                                      batch_size=batch_size,
                                                                      num_workers=0)
    else:
        print("There is no such dataset")

    best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None
    best_acc = None
    all_fscore, all_acc, all_loss = [], [], []

    for e in range(n_epochs):
        start_time = time.time()

        train_loss, train_acc, _, _, _, train_fscore = train_or_eval_model(model, loss_function, kl_loss, train_loader,
                                                                           e, optimizer, True)
        valid_loss, valid_acc, _, _, _, valid_fscore = train_or_eval_model(model, loss_function, kl_loss, valid_loader,
                                                                           e)
        test_loss, test_acc, test_label, test_pred, test_mask, test_fscore = train_or_eval_model(model, loss_function,
                                                                                                 kl_loss, test_loader,
                                                                                                     e)
        all_fscore.append(test_fscore)

        if best_fscore == None or best_fscore <= test_fscore:
            if best_fscore == None:
                best_fscore = test_fscore
                best_acc = test_acc
                best_label, best_pred, best_mask = test_label, test_pred, test_mask

            elif best_acc < test_acc and best_fscore == test_fscore:
                best_acc = test_acc
                best_fscore = test_fscore
                best_label, best_pred, best_mask = test_label, test_pred, test_mask

            elif best_fscore < test_fscore:
                best_acc = test_acc
                best_fscore = test_fscore
                best_label, best_pred, best_mask = test_label, test_pred, test_mask

        if args.tensorboard:
            writer.add_scalar('test: accuracy', test_acc, e)
            writer.add_scalar('test: fscore', test_fscore, e)
            writer.add_scalar('train: accuracy', train_acc, e)
            writer.add_scalar('train: fscore', train_fscore, e)


        print(
            'epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'. \
            format(e + 1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss, test_acc,
                   test_fscore, round(time.time() - start_time, 2)))
        if (e + 1) % 10 == 0:
            print(classification_report(best_label, best_pred, sample_weight=best_mask, digits=4))
            print(confusion_matrix(best_label, best_pred, sample_weight=best_mask))

    if args.tensorboard:
        writer.close()

    print('Test performance..')
    print('F-Score: {}'.format(max(all_fscore)))
    print('F-Score-index: {}'.format(all_fscore.index(max(all_fscore)) + 1))

    if not os.path.exists("record_{}_{}_{}.pk".format(today.year, today.month, today.day)):
        with open("record_{}_{}_{}.pk".format(today.year, today.month, today.day), 'wb') as f:
            pk.dump({}, f)
    with open("record_{}_{}_{}.pk".format(today.year, today.month, today.day), 'rb') as f:
        record = pk.load(f)
    key_ = 'name_'
    if record.get(key_, False):
        record[key_].append(max(all_fscore))
    else:
        record[key_] = [max(all_fscore)]
    if record.get(key_ + 'record', False):
        record[key_ + 'record'].append(classification_report(best_label, best_pred, sample_weight=best_mask, digits=4))
    else:
        record[key_ + 'record'] = [classification_report(best_label, best_pred, sample_weight=best_mask, digits=4)]
    with open("record_{}_{}_{}.pk".format(today.year, today.month, today.day), 'wb') as f:
        pk.dump(record, f)

    print(classification_report(best_label, best_pred, sample_weight=best_mask, digits=4))
    print(confusion_matrix(best_label, best_pred, sample_weight=best_mask))

