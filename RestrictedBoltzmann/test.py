import pickle
import numpy as np
import pandas as pd
from plotnine import * 
from teusday import RBMachine, sampler

P_data = {(-1, -1, -1): 0.25, (-1, 1, 1): 0.25, (1, -1, 1): 0.25, (1, 1, -1): 0.25,
          (-1, 1, -1): 0, (-1, -1, 1): 0, (1, 1, 1): 0, (1, -1, -1): 0}
def p(b):
    """ Sigmoid activation function used for probabilities. """
    return 1 / (1 + np.exp(-2 * b))

def compute_KL_divergence(P_data, P_model):
    """ Function to compute KL divergence between two distributions. """
    D_KL = 0
    for key, P_d in P_data.items():
        if P_d != 0:
            D_KL += P_d * (np.log(P_d) - np.log(P_model[key]))
    return D_KL

def theoretical_upper_bound(N, M):
    if M < 2 ** (N - 1) - 1:
        bound = N - np.log2(M + 1) - ((M + 1) / (2 * np.log2(M + 1)))
    else:
        bound = 0
    return bound

sampler = sampler(n=8)
batch_size = 5000

N = 3
results = []

for i in [1, 2, 4, 8]:
    with open(f"cdk10_trained_rbm_model{i}.pkl", "rb") as f:
        nn = pickle.load(f)
    print(f"The {i}th:")

    data_loader = sampler.sample(batch_size=batch_size)
    for data in data_loader:
        pred = nn.forward(data)

    pattern_count = {(-1, -1, -1): 0, (-1, 1, 1): 0, (1, -1, 1): 0, (1, 1, -1): 0,
                     (-1, 1, -1): 0, (-1, -1, 1): 0, (1, 1, 1): 0, (1, -1, -1): 0}
    
    data_loader = sampler.sample(batch_size=batch_size)
    _, _, preds = nn.forward(data_loader, iter=1000)  # 批量处理整个采样数据

    # statistics
    for pred in preds:
        pattern_tuple = tuple(pred.tolist())
        if pattern_tuple in pattern_count:
            pattern_count[pattern_tuple] += 1

    P_model = {key: count / batch_size for key, count in pattern_count.items()}

    D_KL = compute_KL_divergence(P_data, P_model)
    D_KL_upper = theoretical_upper_bound(N, i)
    print(f"KL divergence for {i} hidden neurons: {D_KL}")
    print(P_model)

    results.append({'Hidden Neurons': i, 'D_KL (Experimental)': D_KL, 'D_KL (Theoretical Upper Bound)': D_KL_upper})

df_results = pd.DataFrame(results)

plot = (
    ggplot(df_results, aes(x='Hidden Neurons'))
    + geom_line(aes(y='D_KL (Experimental)', color='"Experimental"'), size=1.2)
    + geom_point(aes(y='D_KL (Experimental)', color='"Experimental"'), size=3)
    + geom_line(aes(y='D_KL (Theoretical Upper Bound)', color='"Theoretical Upper Bound"'), size=1.2, linetype='dashed')
    + geom_point(aes(y='D_KL (Theoretical Upper Bound)', color='"Theoretical Upper Bound"'), size=3)
    + labs(title="Kullback-Leibler Divergence vs Number of Hidden Neurons",
           x="Number of Hidden Neurons (M)",
           y="D_KL",
           color="Legend")
    + theme_minimal()
)
print(plot)
  