import logging
import torch
logger = logging.getLogger("main")


class Logger(object):
    def __init__(self):
        self.results = []


    def add_result(self, result):
        assert len(result) == 3
        self.results.append(result)


    def print_statistics(self):
        result = 100 * torch.tensor(self.results)
        argmax = result[:, 1].argmax().item()
        logger.info(f'Highest Train: {result[:, 0].max():.2f}')
        logger.info(f'Highest Valid: {result[:, 1].max():.2f}')
        logger.info(f'  Final Train: {result[argmax, 0]:.2f}')
        logger.info(f'   Final Test: {result[argmax, 2]:.2f}')