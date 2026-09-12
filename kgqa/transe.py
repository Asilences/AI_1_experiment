from __future__ import annotations

import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
from pykeen.models import TransE
from pykeen.training import SLCWATrainingLoop
from pykeen.triples import TriplesFactory
import torch

from .common import ROOT, seed_all, setup_logging, write_json


def train(epochs=100):
    seed_all(42)
    output = ROOT / 'models' / 'transe'; output.mkdir(parents=True, exist_ok=True)
    factory = TriplesFactory.from_path(ROOT / 'data' / 'processed' / 'triples.tsv', delimiter='\t')
    model = TransE(triples_factory=factory, embedding_dim=100, scoring_fct_norm=1,
                   loss='MarginRankingLoss', loss_kwargs={'margin': 1.0}, random_seed=42).to('cuda' if torch.cuda.is_available() else 'cpu')
    loop = SLCWATrainingLoop(model=model, triples_factory=factory, optimizer='Adam', optimizer_kwargs={'lr': .001},
                             negative_sampler='basic', negative_sampler_kwargs={'num_negs_per_pos': 5, 'filtered': True})
    losses = loop.train(factory, num_epochs=epochs, batch_size=256, use_tqdm=False, use_tqdm_batch=False)
    np.savez_compressed(output / 'embeddings.npz',
                        entity=model.entity_representations[0](indices=None).detach().cpu().numpy(),
                        relation=model.relation_representations[0](indices=None).detach().cpu().numpy())
    write_json(output / 'mapping.json', {'entity_to_id': factory.entity_to_id, 'relation_to_id': factory.relation_to_id})
    losses = [float(value) for value in losses]; write_json(output / 'history.json', {'loss': losses, 'epochs': epochs})
    plt.figure(figsize=(7, 4)); plt.plot(range(1, len(losses) + 1), losses); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.tight_layout()
    plt.savefig(output / 'loss.png', dpi=180); plt.close(); model.save_state(output / 'model.pkl')
    logging.info('COMPLETE TransE epochs=%s final_loss=%.6f', epochs, losses[-1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--epochs', type=int, default=100); args = parser.parse_args()
    setup_logging('transe'); train(args.epochs)
