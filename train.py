import os
import argparse
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from src.helper_functions.helper_functions import mAP, CocoDetection, CutoutPIL, ModelEma, add_weight_decay, set_random_seeds
from src.models import create_model, to_sdl
from src.loss_functions.SDL_loss import SDLLoss
from randaugment import RandAugment
from torch.cuda.amp import GradScaler, autocast
import pickle
from copy import deepcopy
from src.helper_functions.helper_functions import calc_F1, get_knns

parser = argparse.ArgumentParser(description='Zero shot learning with SDL. MS_COCO Training')
parser.add_argument('--data', metavar='DIR', help='path to dataset', default='/home/MSCOCO_2014/')
parser.add_argument('--metadata', type=str, default='./data/COCO')
parser.add_argument('--lr', default=2.5e-5, type=float)
parser.add_argument('--var_weight', default=0.01, type=float, help='The weight of the regularization of the variance')
parser.add_argument('--num-epochs', type=int, default=10)
parser.add_argument('--model-name', default='tresnet_m')
parser.add_argument('--pretrain-backbone', type=int, default=0, help='Use a pre-trained recognition backbone')
parser.add_argument('--model-path', default='./tresnet_m.pth', type=str)
parser.add_argument('--num-classes', default=1000, help='pretrain backbone num classes')
parser.add_argument('--autocast-enabled', type=int, default=1)
parser.add_argument('--num_rows', type=int, default=2)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--wordvec_dim', type=int, default=300)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--image-size', default=224, type=int,
                    metavar='N', help='input image size (default: 224)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('--path_output', type=str, default='./outputs')


def main():
    args = parser.parse_args()
    args.do_bottleneck_head = False

    if args.seed is not None:
        set_random_seeds(args.seed)
    # Setup model
    print('creating model...')
    model = create_model(args).cuda()
    if args.model_path:  # make sure to load pretrained ImageNet model
        state = torch.load(args.model_path, map_location='cpu')
        filtered_dict = {k: v for k, v in state['model'].items() if
                         (k in model.state_dict() and 'head.fc' not in k)}
        model.load_state_dict(filtered_dict, strict=False)
        if args.pretrain_backbone:  # if we start from a backbone need to modify the model
            model = to_sdl(model, args)
        print('done\n')

    # COCO Data loading
    instances_path_val = os.path.join(args.metadata, 'zs_split/val_17_48.json')
    instances_path_val_unseen = os.path.join(args.metadata, 'zs_split/val_unseen.json')
    instances_path_train = os.path.join(args.metadata, 'zs_split/train_17_48.json')
    data_path_val = f'{args.data}/'  # args.data
    data_path_train = f'{args.data}/'  # args.data
    val_dataset = CocoDetection(data_path_val,
                                instances_path_val,
                                transforms.Compose([
                                    transforms.Resize((args.image_size, args.image_size)),
                                    transforms.ToTensor(),
                                    # normalize, # no need, toTensor does normalization
                                ]))
    val_unseen_dataset = CocoDetection(data_path_val,
                                       instances_path_val_unseen,
                                       transforms.Compose([
                                           transforms.Resize((args.image_size, args.image_size)),
                                           transforms.ToTensor(),
                                           # normalize, # no need, toTensor does normalization
                                       ]))
    train_dataset = CocoDetection(data_path_train,
                                  instances_path_train,
                                  transforms.Compose([
                                      transforms.Resize((args.image_size, args.image_size)),
                                      CutoutPIL(cutout_factor=0.5),
                                      RandAugment(),
                                      transforms.ToTensor(),
                                      # normalize,
                                  ]))
    print("len(val_dataset)): ", len(val_dataset))
    print("len(train_dataset)): ", len(train_dataset))

    # Pytorch Data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    val_unseen_loader = torch.utils.data.DataLoader(
        val_unseen_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    with open(os.path.join(args.metadata, "wordvec_array.pickle"), 'rb') as fp:
        wordvec_array = pickle.load(fp)
    wordvec_array = wordvec_array['wordvec_array']
    cls_ids = pickle.load(open(os.path.join(args.metadata, "cls_ids.pickle"), "rb"))
    if not os.path.isdir(args.path_output):
        os.makedirs(args.path_output)
    # Actual Training
    train_zsl(model, train_loader, val_loader, val_unseen_loader, wordvec_array, args,
              unseen_ids=cls_ids['test'], seen_ids=cls_ids['train'])


def train_zsl(model, train_loader, val_loader, val_unseen_loader, wordvec_array, args, unseen_ids=None, seen_ids=None):
    lr = args.lr
    enabled = args.autocast_enabled
    # ema = ModelEma(model, 0.9997)  # 0.9997^641=0.82 #Todo: enable ema as an option (not used in the paper)

    # set optimizer
    Epochs = args.num_epochs
    Stop_epoch = args.num_epochs
    weight_decay = 3e-4
    seen_ids_tensor = torch.tensor(list(seen_ids)).cuda()
    wordvec_array = torch.tensor(wordvec_array).cuda().float()
    seen_wordvec = deepcopy(wordvec_array)
    seen_wordvec = seen_wordvec[:, :,
                   list(seen_ids)]  # use only seen tags
    criterion = SDLLoss(wordvec_array=seen_wordvec,
                        weight=args.var_weight) 

    parameters = add_weight_decay(model, weight_decay)
    optimizer = torch.optim.Adam(params=parameters, lr=lr, weight_decay=0)  # true wd, filter_bias_and_bn
    steps_per_epoch = len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=Epochs,
                                        pct_start=0.1)
    highest_F1 = 0
    highest_F1_unseen = 0
    trainInfoList = []
    scaler = GradScaler(enabled=enabled)
    for epoch in range(Epochs):
        if epoch > Stop_epoch:
            break
        for i, (inputData, target) in enumerate(train_loader):
            inputData = inputData.cuda()
            target = target.cuda()  # (batch,3,num_classes)
            target = target.max(dim=1)[0]
            target = target[:, seen_ids_tensor]  # use only seen
            with autocast(enabled=bool(enabled)):  # mixed precision
                output = model(inputData).float()
            loss = criterion(output, target)
            model.zero_grad()

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            scheduler.step()

            # ema.update(model)
            # store information
            if i % 100 == 0:
                trainInfoList.append([epoch, i, loss.item()])
                print('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f}'
                      .format(epoch+1, Epochs, str(i).zfill(3), str(steps_per_epoch).zfill(3),
                              scheduler.get_last_lr()[0], \
                              loss.item()))

        try:
            torch.save(model.state_dict(), os.path.join(
                args.path_output, 'model-{}-{}.ckpt'.format(epoch + 1, i + 1)))
        except:
            pass

        model.eval()
        seen_and_unseen = seen_ids | unseen_ids
        F1_score = validate_multi(val_loader, model, wordvec_array[:, :, list(seen_and_unseen)],
                                  relevant_ids=list(seen_and_unseen))
        F1_score_unseen = validate_multi(val_unseen_loader, model, wordvec_array[:, :, list(unseen_ids)],
                                         relevant_ids=list(unseen_ids))

        model.train()
        # Save model based on a specific metric
        if F1_score > highest_F1:
            highest_F1 = F1_score
            try:
                torch.save(model.state_dict(), os.path.join(
                    args.path_output, 'model-highest.ckpt'))
            except:
                pass
        if F1_score_unseen > highest_F1_unseen:
            highest_F1_unseen = F1_score_unseen
        # showing the highest F1 for generalized and zero shot over different epochs
        print('Generalized:: current_F1 = {:.2f}, highest_F1 = {:.2f}\n'.format(F1_score, highest_F1))
        print('Zero-shot:: current_F1 = {:.2f}, highest_F1 = {:.2f}\n'.format(F1_score_unseen, highest_F1_unseen))


def validate_multi(val_loader, model, word_vecs, relevant_ids=None, top_k=3):
    print("starting validation")
    word_vecs = word_vecs.squeeze().transpose(0, 1)
    preds_regular = []
    preds_ema = []
    targets = []
    for i, (input, target) in enumerate(val_loader):
        target = target
        target = target.max(dim=1)[0]
        # compute output
        with torch.no_grad():
            with autocast():
                output_regular = model(input.cuda()).cpu()
                # output_ema = ema_model.module(input.cuda()).cpu()

        # for metrics calculation
        preds_regular.append(output_regular.cpu().detach())
        # preds_ema.append(output_ema.cpu().detach())
        targets.append(target.cpu().detach())

    idxs, dists = get_knns(word_vecs.cpu().detach(), torch.cat(preds_regular).numpy())

    precision_3, recall_3, F1_3 = calc_F1(torch.cat(targets).numpy(), idxs, top_k, relevant_inds=relevant_ids,
                                          num_classes=len(word_vecs))
    if F1_3 != F1_3:
        F1_3 = 0
    print("Top-{}: precision {:.2f}, recall {:.2f}, F1 {:.2f}".format(top_k, precision_3, recall_3, F1_3))

    dists = get_knns(torch.cat(preds_regular).numpy(), word_vecs.cpu().detach(), for_map=True)
    mAP_score_regular = mAP(torch.cat(targets).numpy(), dists.transpose(), relevant_inds=relevant_ids,
                            num_classes=len(word_vecs))
    print("mAP score  {:.2f}".format(mAP_score_regular))

    # mAP_score_ema = mAP(torch.cat(targets).numpy(), torch.cat(preds_ema).numpy())
    # print("mAP score regular {:.2f}, mAP score EMA {:.2f}".format(mAP_score_regular, mAP_score_ema))
    return F1_3


if __name__ == '__main__':
    main()
