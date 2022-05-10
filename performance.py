import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np

class Evaluate:
    def __init__(self, path: str):
        with open(path) as json_dict:
            self.json_dict = json.load(json_dict)

        self.all_train_loss = self.json_dict['all_training_loss']
        self.all_val_loss = self.json_dict['all_val_loss']
        self.test_acc_top1 = self.json_dict['test_acc_top1']
        self.all_val_acc_top1 = self.json_dict['all_val_acc_top1']

    def compare_top1_test_acc(self,  path2, suffix_1='_fft', suffix_2='_grey'):

        with open(path2) as acc_n_loss:
            stats2 = json.load(acc_n_loss)
            test_acc_top2 = stats2['test_acc_top1']
        if self.test_acc_top1.keys() != test_acc_top2.keys():
            raise ValueError('Two results to compare must have same keys!')
        for key, val in self.test_acc_top1.items():

            print(f"training sequence: {key}")
            result1 = pd.DataFrame(val, columns=['Test set distance', 'top1 acc'])
            result2 = pd.DataFrame(test_acc_top2[key],columns=['Test set distance', 'top1 acc'])
            merged = result1.merge(result2, on='Test set distance', suffixes=(suffix_1, suffix_2))
            print(merged)

    def top1_test_acc_to_pd(self):
        col = []
        for k, v in self.test_acc_top1.items():
            df_col = pd.DataFrame(v,columns=['Test set distance', k])
            df_col.set_index('Test set distance', drop=True, inplace=True)
            col.append(v)
        result = pd.concat(col, axis=1)
        return result

    def plot_loss(self, early_stop: bool = False):
        fig, axs = plt.subplots(2, 1)

        plt.rcParams['figure.figsize'] = [10, 10]

        target_sequence = '[0.2, 0.4, 0.6, 0.8, 1]'

        x_start = 0
        color_palette = ['#fe4a49', '#2ab7ca', '#fed766', '#e6e6ea', '#f4f4f8']
        color_palette2 = ['#e3c9c9', '#f4e7e7', '#eedbdb', '#cecbcb', '#cbdadb']
        grey_color = [(153, 153, 153), (119, 119, 119), (85, 85, 85), (51, 51, 51), (17, 17, 17)]
        for idx, i in enumerate(self.all_val_loss[target_sequence]):
            x_end = x_start + len(i[1]['1'])
            x = np.arange(x_start, x_end)
            num_distances = len(i[1]) - 1
            count = 0
            train_loss_list = [t[1] for t in self.all_train_loss[target_sequence] if t[0] == i[0]][0]
            axs[0].plot(x, train_loss_list, color=(0, 0, 1, 0.5))
            min_loss_idx = i[1]['avg'].index(min(i[1]['avg']))
            for val_distance, loss in i[1].items():

                if val_distance == 'avg':
                    if early_stop:
                        axs[1].plot(x[:min_loss_idx], loss[:min_loss_idx], color='red')
                    else:
                        axs[1].plot(x, loss, color='red')
                        axs[1].plot(x_start + min_loss_idx, min(loss), '*g')
                else:
                    color = (num_distances - count) / num_distances
                    # color = 'black'
                    color = grey_color[count]
                    color = (color[0] / 255, color[1] / 255, color[2] / 255, 0.3)
                    if early_stop:
                        axs[1].plot(x[:min_loss_idx], loss[:min_loss_idx], color=color)
                    else:
                        axs[1].plot(x, loss, color=color)
                count += 1

            if early_stop:
                axs[0].axvspan(x_start, x_start + min_loss_idx, color=color_palette2[idx], label=i[0])
                axs[1].axvspan(x_start, x_start + min_loss_idx, color=color_palette2[idx], label=i[0])

            axs[0].axvspan(x_start, x_end, color=color_palette2[idx], label=i[0])
            axs[1].axvspan(x_start, x_end, color=color_palette2[idx], label=i[0])
            x_start = x_end + 10
        # fig.tight_layout()
        plt.legend()

        plt.show()




"""
test_acc_top1
[1, 0.8, 0.6, 0.4, 0.2] [['0.2', 0.47593582887700536], ['0.4', 0.40488922841864017], ['0.6', 0.32314744079449964], ['0.8', 0.2696715049656226], ['1', 0.22077922077922077], ['mixed', 0.3254392666157372]]
[0.8, 0.6, 0.4, 0.2] [['0.2', 0.48357524828113063], ['0.4', 0.41940412528647825], ['0.6', 0.29258976317799845], ['0.8', 0.20244461420932008], ['1', 0.1711229946524064], ['mixed', 0.30786860198624905]]
[0.6, 0.4, 0.2] [['0.2', 0.48968678380443087], ['0.4', 0.3865546218487395], ['0.6', 0.26585179526355995], ['0.8', 0.18792971734148206], ['1', 0.1627196333078686], ['mixed', 0.28113063407181055]]
[0.4, 0.2] [['0.2', 0.48357524828113063], ['0.4', 0.34988540870893814], ['0.6', 0.23529411764705882], ['0.8', 0.1573720397249809], ['1', 0.11459129106187929], ['mixed', 0.2643239113827349]]
[0.2] [['0.2', 0.47364400305576776], ['0.4', 0.2979373567608862], ['0.6', 0.186401833460657], ['0.8', 0.1573720397249809], ['1', 0.13292589763177998], ['mixed', 0.23300229182582124]]
"""

def visualize(path):
  with open(path) as acc_n_loss:
    stats = json.load(acc_n_loss)
    test_acc_top1 = stats['test_acc_top1']
    for key, val in test_acc_top1.items():
      print(f"training sequence: {key}")
      result = pd.DataFrame(val, columns = ['Test set distance', 'top1 acc'])
      result.index = result['Test set distance']
      result.drop(columns=['Test set distance'],inplace=True)
      print(result)
      print("\n")



