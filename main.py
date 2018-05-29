from pagerank import pagerank
import argparse


def main():
        parser = argparse.ArgumentParser(description='Running PageRank')
        parser.add_argument('path_file', metavar='f', type=str,
                            help='path to file to perform page ranking')
        parser.add_argument('beta', metavar='b', type=float, help='allow teleportation with 1-beta probability')
        parser.add_argument('eps', metavar='eps', type=float, help='epsilon value for convergence')
        parser.add_argument('n_nodes', metavar='nodes', type=int, help='number of unique nodes')
        parser.add_argument('n_edges', metavar='edges', type=int, help='number of all edges')
        parser.add_argument('pow_iter', metavar='pow_iter', type=int, help='number of iterations')

        args = parser.parse_args()
        file_path = args.path_file
        beta = args.beta
        eps = args.eps
        n_nodes = args.n_nodes
        n_edges = args.n_edges
        num_iterations = args.pow_iter

        result = pagerank(file_name=file_path, beta=beta,power_iterations=num_iterations,
                 num_edges=n_edges, num_nodes=n_nodes, eps=eps)

if __name__ == '__main__':
    main()