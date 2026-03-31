import os
import pickle
import torch
from RatingSystem import RatingSystem
from solution import EnhancedRecommender, CONFIG

class MySystem(RatingSystem):
    _model = None
    _user2idx = None
    _movie2idx = None

    def __init__(self):
        super().__init__()

    @classmethod
    def _load_model(cls):
        if cls._model is not None:
            return
        if not os.path.exists('model.pth') or not os.path.exists('user2idx.pkl') or not os.path.exists('movie2idx.pkl'):
            raise FileNotFoundError('model.pth and mapping files must exist, run solution.py training first')
        with open('user2idx.pkl', 'rb') as f:
            cls._user2idx = pickle.load(f)
        with open('movie2idx.pkl', 'rb') as f:
            cls._movie2idx = pickle.load(f)
        model = EnhancedRecommender(len(cls._user2idx), len(cls._movie2idx), CONFIG['embedding_dim'])
        model.load_state_dict(torch.load('model.pth', map_location=CONFIG['device']))
        model.to(CONFIG['device'])
        model.eval()
        cls._model = model

    def rate(self, user, movie_id):
        """Zwraca prognozowaną ocenę w skali 1-5, używając uprzednio wytrenowanego modelu."""
        if movie_id in user.ratings:
            return user.ratings[movie_id]
        try:
            self._load_model()
        except Exception:
            return 2.5

        u_idx = self._user2idx.get(user.id)
        m_idx = self._movie2idx.get(movie_id)
        if u_idx is None or m_idx is None:
            return 2.5

        with torch.no_grad():
            u_t = torch.tensor([u_idx], dtype=torch.long, device=CONFIG['device'])
            m_t = torch.tensor([m_idx], dtype=torch.long, device=CONFIG['device'])
            out = self._model(u_t, m_t).item()
        return float(out * 4 + 1)  # Denormalizacja na 1-5

    def __str__(self):
        return 'System created by 111333 and 333444'
