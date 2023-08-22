import argparse
import os

import torch

from dt_utils import env_from_model_path, load_model, load_data_compute_fisher, get_save_path

def main(args):
    env = env_from_model_path(args.model, args)

    # load model
    model = load_model(args.model, device=args.device)
    batch_size = args.f_batch_size
    num_batches = args.f_num_batches
    save_path = get_save_path(args.model, batch_size, num_batches)

    # check if save_name exists
    if os.path.exists(save_path):
        print(f'{save_path} already exists')
        return

    # load data and compute fisher information
    info = load_data_compute_fisher(model, env, args)

    # save fisher information
    torch.save(info, save_path)
    print("saved at", save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./data', type=str)
    parser.add_argument('--model', type=str, required=True)

    # Settings for computing fisher information
    parser.add_argument('--f_batch_size', default=64, type=int)
    parser.add_argument('--f_num_batches', default=100, type=int)

    parser.add_argument('--K', default=20, type=int)

    parser.add_argument("--device", help="PyTorch device to be use (ex: cpu, cuda...)", default="cuda:0", type=str)

    args = parser.parse_args()
    main(args)