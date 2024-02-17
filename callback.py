from stable_baselines3.common.callbacks import BaseCallback

class PrintInfoCallback(BaseCallback):
    def _on_step(self) -> bool:
        # Access the 'infos' for the last environment step
        # 'infos' is a list of dictionaries for each environment in the vectorized setup
        infos = self.locals['infos']
        
        # Example: Print 'infos' of the first environment in the vector
        if infos and 'ALC' in infos[0]:
            print(f"ALC: {infos[0]['ALC']}, AIR: {infos[0]['AIR']}, ARL: {infos[0]['ARL']}")
        
        return True
