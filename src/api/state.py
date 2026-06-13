from config.settings import PRODUCTION_MODEL

class AppState:
    def __init__(self):
        self.current_model: str = PRODUCTION_MODEL

# Global application state
state = AppState()
