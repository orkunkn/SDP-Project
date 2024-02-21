from stable_baselines3.common.callbacks import BaseCallback

# A class used for printing info during agent working
class PrintInfoCallback(BaseCallback):
    def _on_step(self) -> bool:
        infos = self.locals['infos']
        
        if infos and 'ALC' in infos[0]:
            print(infos[0])
        
        return True
