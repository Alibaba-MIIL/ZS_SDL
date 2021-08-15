import torch
from src.models import create_model
import argparse
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pickle
from src.plot_results import plot_results
from src.loss_functions.SDL_loss import SDLLoss


# ----------------------------------------------------------------------
# Parameters
parser = argparse.ArgumentParser(description='Zero shot learning with SDL pretrained model.')
parser.add_argument('--model_path', type=str, default='./models_local/NUS_mtresnet_224.pth')
parser.add_argument('--pic_path', type=str, default='./pics/140016_215548610_422b79b4d7_m.jpg')
parser.add_argument('--model_name', type=str, default='tresnet_m')
parser.add_argument('--num_rows', type=int, default=7)
parser.add_argument('--pretrain-backbone', type=int, default=0)
parser.add_argument('--wordvec_dim', type=int, default=300)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--dataset_path', type=str, default='./data/NUS_WIDE')
parser.add_argument('--dataset_type', type=str, default='NUS_WIDE')
parser.add_argument('--path_output', type=str, default='./outputs')
parser.add_argument('--top_k', type=int, default=10)


def main():
    print('ZSL demo of inference code on a single image.')

    # ----------------------------------------------------------------------
    # Preliminaries
    args = parser.parse_args()

    # setup model
    print('creating and loading the model...')
    state = torch.load(args.model_path, map_location='cpu')
    args.num_classes = state['num_classes']
    model = create_model(args).cuda()
    model.load_state_dict(state['model'], strict=True)
    model.eval()
    classes_list = np.array(list(state['idx_to_class'].values()))
    print('done\n')

    # setup data
    with open(os.path.join(args.dataset_path, "classes.pickle"), 'rb') as fp:
        classes = pickle.load(fp)
    with open(os.path.join(args.dataset_path, "unseen_classes.pickle"), 'rb') as fp:
        unseen_classes = pickle.load(fp)
    with open(os.path.join(args.dataset_path, "wordvec_array.pickle"), 'rb') as fp:
        wordvec_array = pickle.load(fp)

    # ----------------------------------------------------------------------
    # Inference
    print('loading image and doing inference...')
    im = Image.open(args.pic_path)
    im_resize = im.resize((args.input_size, args.input_size))
    np_img = np.array(im_resize, dtype=np.uint8)
    tensor_img = torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0  # HWC to CHW
    tensor_batch = torch.unsqueeze(tensor_img, 0).cuda()
    output = model(tensor_batch)
    np_output = output.cpu().detach().numpy()
    print('done\n')

    # Extract predicted tags
    x = np_output.reshape(args.wordvec_dim, args.num_rows)
    p = np.dot(x.T, wordvec_array)
    dist = -np.max(p, axis=0)
    sorted_ids = np.argsort(dist)
    top_k_ids = sorted_ids[:args.top_k]
    pred_classes = [classes[i] for i in top_k_ids]

    # Visualize results
    img_result = plot_results(im_resize, pred_classes, classes, unseen_classes, args.path_output,
                              os.path.split(args.pic_path)[1])

    # ----------------------------------------------------------------------
    # Example of loss calculation
    output = model(tensor_batch)
    loss_func = SDLLoss(wordvec_array=torch.tensor(wordvec_array).unsqueeze(0))
    batch_siz = 1
    target = torch.zeros(batch_siz, args.num_classes)
    target[0, 40] = 1
    loss = loss_func.forward(output.cpu(), target.cpu())
    print("Example loss: %0.2f" % loss)
    print('Done\n')


if __name__ == '__main__':
    main()
