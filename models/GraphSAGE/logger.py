import torch


class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            decision_idx = 0  # TODO need valid
            argmax = result[:, decision_idx].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train ({result[:, 0].argmax().item()}): {result[:, 0].max():.2f}')
            print(f'Highest Valid ({result[:, 1].argmax().item()}): {result[:, 1].max():.2f}')
            print(f'Highest Test ({result[:, 2].argmax().item()}): {result[:, 2].max():.2f}')
            print(f'  Final Train ({argmax}): {result[argmax, 0]:.2f}')
            print(f'   Final Test ({argmax}): {result[argmax, 2]:.2f}')
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            decision_idx = 0  # TODO need valid
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                test1 = r[:, 2].max().item()
                train2 = r[r[:, decision_idx].argmax(), 0].item()
                test = r[r[:, decision_idx].argmax(), 2].item()
                best_results.append((train1, valid, test1, train2, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'Highest Test: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 4]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')
