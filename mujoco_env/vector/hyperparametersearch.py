""" search for the optimal hyperparameter values  to run several in parallel use different """
import os
import argparse


def main(args):
    """ try different values for epsilon and the decay
        and the two parameter from per alpha and beta
    Args:
        param1 (args) :
    """
    
    list_of_update_freq= [15, 20, 2, 5]
    for freq in list_of_update_freq:
        freq = int(freq)
        os.system(f'python3 ./main_parallel.py \
                --target_update_freq {freq}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Average_TD3')
    main(parser.parse_args())
