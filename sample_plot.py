from calculate_distances import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_useen_clients_cdf(file_name,num_seen = 500,num_total = 1000,n_k = 1000,delta = 0.05,epsilons = {"chi":0.2,"kl":0.2}): 
  network_file = pd.read_csv(file_name)
  acc_total = network_file['accuracy'].values
  loss_total = 1-acc_total
  acc_seen = acc_total[:num_seen]
  loss_seen = loss_total[:num_seen]

  K = len(acc_seen)
  constant = np.sqrt(np.log((K + 2) / delta) / (2 * n_k))
  min_l, max_l = min(loss_seen), max(loss_seen)
  lambdas = np.linspace(min_l + constant, max_l + constant, 50)

  for f_divergence in ["chi","kl"]:
    if(f_divergence == "chi"):
        probs =  np.array([calcualte_fdivergence_optimization(loss_seen,lambda_,epsilons[f_divergence],'chi-square') for lambda_ in lambdas]) +  np.sqrt((np.log((K) / delta)) / K)
    if(f_divergence == "kl"):
        probs = np.array([calcualte_fdivergence_optimization(loss_seen,lambda_,epsilons[f_divergence],'kl') for lambda_ in lambdas]) + np.sqrt((np.log((K) / delta)) / K)
    probs = np.clip(probs,0,1)
    plt.plot(1-lambdas,probs, linestyle='--',label=f'$d_{{{f_divergence}}}={round(epsilons[f_divergence], 3)}$',linewidth=2.4)

  n_total = len(loss_total)
  plt.plot(np.sort(1-loss_total), np.arange(1,n_total + 1) / n_total, linestyle='-',label=f'$Meta$',linewidth=2.4)
  plt.xlabel('accuracies')
  plt.ylabel('Cumulative Probability')
  plt.legend()
  plt.show()
  #plt.savefig("seen.png")

def plot_meta_fdivergence(file_name1,file_name2,f,epsilons,delta = 0.05,n_k = 1000):
  network_file1 = pd.read_csv(file_name1)
  meta1_accs = network_file1['accuracy'].values
  meta1_loss = 1 - meta1_accs
  K = len(meta1_loss)
  constant = np.sqrt(np.log((K + 2) / delta) / (2 * n_k))
  network_file2 = pd.read_csv(file_name2)
  meta2_accs = network_file2['accuracy'].values
  meta2_loss = 1 - meta2_accs

  plt.figure(figsize=(10,5))
  n = len(meta1_accs)

  for dist in ['Source','Target']:
      if(dist == 'Source'):
        acc_data = meta1_accs

      if(dist == 'Target'):
        acc_data = meta2_accs

      n = len(acc_data)
      plt.plot(np.sort(acc_data),np.arange(1,n + 1) / n, linestyle='-',label=f'${{{dist}}}$',linewidth=2.4)

  min_l, max_l = min(meta1_loss), max(meta1_loss)
  lambdas = np.linspace(min_l + constant, max_l + constant, 50)
  for epsilon in epsilons:
    n = len(meta1_accs)
    if(f == "chi"):
      probs =  np.array([calcualte_fdivergence_optimization(meta1_loss,lambda_,epsilon,'chi-square') for lambda_ in lambdas]) +  np.sqrt((np.log((K) / 0.05)) / K)
    if(f == "kl"):
      probs = np.array([calcualte_fdivergence_optimization(meta1_loss,lambda_,epsilon,'kl') for lambda_ in lambdas]) + np.sqrt((np.log((K) / 0.05)) / K)

    probs = np.clip(probs,0,1)
    plt.plot(1-lambdas,probs, linestyle='--',label=f'$d_{{{f}}}={round(epsilon, 3)}$',linewidth=2.4,color = 'red')

    ## calculate two meta-distribution distance
    if(f == 'chi'):
        div = calculate_meta_chisquare_divergence(meta2_accs, meta1_accs,10)
    if(f == 'kl'):
        div = calculate_meta_kl_divergence(meta2_accs, meta1_accs, bins=10)
    plt.xlabel('Accuracy')
    plt.ylabel('Cumulative Probability')
    plt.title(f'{f} divergence between Source and Target Meta Distributions: {div:.3f}')
    plt.legend()
    plt.show()
    #plt.savefig("meta.png")

if __name__ == "__main__":
    unseen_client_file_name = 'YOUR CDF CSV FILE'
    plot_useen_clients_cdf(unseen_client_file_name)
    meta1_file_name = 'YOUR CDF CSV FILE'
    meta2_file_name = 'YOUR CDF CSV FILE'
    plot_meta_fdivergence(meta1_file_name,meta2_file_name,"chi",[0.3])