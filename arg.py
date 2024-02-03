import argparse
def get_command_line_args():
    parser = argparse.ArgumentParser(description='SNS')
    parser.add_argument('--dataset', default='cora', type=str,choices=["cora", "pubmed", "citeseer" ,"arxiv","product"])
    parser.add_argument('--mode', default='tl', type=str,choices=['t','l','tl',"none"])
    parser.add_argument('--k', type=int, default=-1)
    #gamma and upper limit for each layer
    args = parser.parse_args()
    return args