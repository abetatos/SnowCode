from typing import List
from abc import abstractmethod, ABC
import matplotlib.pyplot as plt


class HarryPlotter: 

    @property
    @abstractmethod
    def FEATURES(self):
        return list

    @property
    def N_FEATURES(self):
        return len(self.FEATURES)
    
    def plot_inputs(self, inputs, item = 0):
        fig, axs = plt.subplots(1, self.N_FEATURES, figsize=(11, 3))
        for i, feature in enumerate(self.FEATURES):
            axs[i].set_xlabel(feature)

        aux = inputs[item]
        for ax, bool_ in zip(axs, self.plot_bool):
            if bool_:
                if len(aux.shape) == 2:
                    ax.imshow(aux.cpu())
                else:
                    ax.imshow(aux[0].cpu())
                    aux = aux[1:,]
        plt.show()

    @classmethod
    def plot_heatmap(self, labels, outputs, mask, item=0):
        delta = labels[item] - outputs[item]
        delta[~mask[item].bool()] = 0
        delta = delta.abs().detach().cpu().numpy().squeeze()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(5, 5))
        ax1.set_xlabel("Error heatmap")
        ax1.imshow(delta, cmap="RdYlGn_r")
        fig.colorbar(ax1.pcolormesh(delta, cmap="RdYlGn_r"))
        plt.show()

    @classmethod
    def plot_outputs(self, labels, outputs, item=0):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 3))
        ax1.set_xlabel("Target")
        ax2.set_xlabel("Prediction")

        target = labels[item].squeeze().cpu()
        preds = outputs[item].detach().cpu().numpy().squeeze()
        ax1.imshow(target)
        ax2.imshow(preds)

        fig.colorbar(ax1.pcolormesh(target))
        fig.colorbar(ax2.pcolormesh(preds))
        
        plt.show()

    @classmethod
    def plot_evolution(self, glob_loss: List[float], glob_val: List[float], title: str):
        fig, ax1 = plt.subplots()
        # Plot linear sequence, and set tick labels to the same color
        
        ax1.plot(range(len(glob_val)), glob_val, "-", color='green', label="validation")
        ax1.tick_params(axis='y', labelcolor='green')
        ax1.set_title(title)
        # Generate a new Axes instance, on the twin-X axes (same position)
        ax2 = ax1.twinx()
        # Plot exponential sequence, set scale to logarithmic and change tick color
        ax2.plot(range(len(glob_loss)), glob_loss, "-", color='red', label="loss")
        ax2.tick_params(axis='y', labelcolor='red')

        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        plt.legend()
        plt.show()