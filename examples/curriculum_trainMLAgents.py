"""
Script for automating curriculum training
"""
import yaml
import sys
import argparse
import os
import glob


def train(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    trainer_config_path = os.path.join(args.agent_path, 'data/trainer_config.yaml')
    arena_config_paths = sorted(glob.glob(os.path.join(args.agent_path, 'data/*')))
    arena_config_paths = [folder for folder in arena_config_paths if os.path.isdir(folder)]
    for idx, arena_config_path in enumerate(arena_config_paths):
        suffix = os.path.basename(arena_config_path)
        print('\n'*5)
        print('\tTraining with arena config: %s' % os.path.basename(suffix))
        command = "python trainMLAgents.py %s %s" % (arena_config_path, trainer_config_path)
        command += ' --n_envs %i' % args.n_envs
        command += ' --n_arenas %i' % get_n_arenas(arena_config_path)
        command += ' --save_freq %i' % args.save_freq
        command += ' --keep_checkpoints %i' % args.keep_checkpoints
        command += ' --suffix %s' % suffix
        # TODO: add more flexibility for model loading and resetting
        if idx:
            command += ' --load_model'
            command += ' --reset_steps'
        ret = os.system(command)
        if ret:
            break
        # copy the model for the next training
        if idx < len(arena_config_paths) - 1:
            model_in = '%s_%s' % (os.path.basename(args.agent_path), suffix)
            model_out = '%s_%s' % (os.path.basename(args.agent_path), os.path.basename(arena_config_paths[idx+1]))
            copy_model_for_next_train(model_in, model_out)

def get_n_arenas(arena_config_path):
    # TODO: detect automatically the number of arenas
    return 4

def copy_model_for_next_train(model_in, model_out):
    command = 'cp -r models/%s models/%s' % (model_in, model_out)
    print(command)
    os.system(command)
    model_filepaths = sorted(glob.glob('models/%s/Learner/model-*' % model_out))
    steps = [int(os.path.basename(filepath).split('model-')[-1].split('.cptk')[0]) for filepath in model_filepaths]
    max_steps = max(steps)
    if len(model_filepaths) > 3:
        for model_filepath, step in zip(model_filepaths, steps):
            if step < max_steps:
                print('Deleting: %s' % model_filepath)
                os.remove(model_filepath)


def parse_args(args):
    epilog = """
    python curriculum_trainMLAgents.py /media/guillermo/Data/Kaggle/animalai/agents/012_even_more_memory --n_envs 4
    """
    description = """
    Script for automating curriculum training
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=epilog)
    parser.add_argument('agent_path', help='Path to the folder of the agent')
    parser.add_argument('--n_envs', type=int, default=4, help='Number of environments to run')
    parser.add_argument('--save_freq', type=int, default=10000, help='Number of steps between each saving of the model.')
    parser.add_argument('--keep_checkpoints', type=int, default=10, help='Number of checkpoints that will be kept.')
    return parser.parse_args(args)

if __name__ == '__main__':
    train()