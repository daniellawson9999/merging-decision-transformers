import os
import itertools

device = 'cuda:2'

n_episodes = 50
max_iters = 10
steps_per_iter = 10000

def get_name_dataset(model_name):
    split_name = model_name.split('-')
    env = split_name[0]
    dataset = split_name[1:-1][0]
    return env, dataset

def mergeff_models(model_1, model_2, model_3):
    env_1, dataset_1 = get_name_dataset(model_1)
    env_2, dataset_2 = get_name_dataset(model_2)
    env_3, dataset_3 = get_name_dataset(model_3)
    command = f'python experiment.py -w --env {env_1} --dataset {dataset_1} --pretrained_models ./models/{env_1}/dt_gym-experiment-{model_1}_best.pt ./models/{env_2}/dt_gym-experiment-{model_2}_best.pt ./models/{env_3}/dt_gym-experiment-{model_3}_best.pt --num_eval_episodes={n_episodes} --max_iters={max_iters} --num_steps_per_iter={steps_per_iter} --include attn mlp --device={device} --copy_model --merge_frozen'

    print("executing", command)
    os.system(command)

if __name__ == '__main__':
    combinations = [
        ['halfcheetah-medium-681135', 'walker2d-medium-876576', 'hopper-medium-599994'],
        ['walker2d-medium-876576', 'hopper-medium-599994', 'halfcheetah-medium-681135'],
        ['hopper-medium-599994', 'halfcheetah-medium-681135', 'walker2d-medium-876576']
    ]
    for triple in combinations:
        model_1 = triple[0]
        model_2 = triple[1]
        model_3 = triple[2]
        mergeff_models(model_1, model_2, model_3)
