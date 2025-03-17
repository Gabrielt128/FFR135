import numpy as np
import pickle

class RBMachine:
  def __init__(self, M) -> None:
    self.M = M
    self.weight = np.random.normal(size = (3, self.M)) 
    self.theta1 = np.zeros(shape = (3))
    self.theta2 = np.zeros(shape = (self.M))

  def __getitem__(self, key):
    return self.__dict__[key]

  def __setitem__(self, key, value):
    if key in self.__dict__:
        self.__dict__[key] = value
    else:
        raise KeyError(f"Key '{key}' not found in parameters.")
      
  @staticmethod
  def stochastic_mcp(b):
    # vectorize this method
    prob_b = 1 / (1 + np.exp(-2*b))
    return 2 * np.random.binomial(n=1, p=prob_b) - 1

  def forward(self, x, iter = 100):
    b0 = np.dot(x, self.weight) - self.theta2
    v = x.copy()
    for i in range(iter):
      raw_h = np.dot(v, self.weight) - self.theta2
      h = RBMachine.stochastic_mcp(raw_h)

      raw_v = np.dot(h, self.weight.T) - self.theta1
      v = RBMachine.stochastic_mcp(raw_v)
    return b0, raw_h, v  
  
class sampler:
  def __init__(self, n=4) -> None:
    self.n = n
    self.pool1 = np.array([[-1, -1, -1], [1, 1, -1], [-1, 1, 1], [1, -1, 1]])
    self.pool2 = np.array([[-1, -1, -1], [1, 1, -1], [-1, 1, 1], [1, -1, 1],
                            [-1, -1, 1], [1, 1, 1], [-1, 1, -1], [1, -1, -1]])
    
  def sample(self, batch_size=32):
      # something intertsted np array = ([]) not () nor []
      if self.n == 4:
          indexes = np.random.choice(4, batch_size) # sample from 4 patterns 
          return self.pool1[indexes]
      elif self.n == 8:
          indexes = np.random.choice(8, batch_size)  # sample from 8 patterns
          return self.pool2[indexes]

def train(rbm, sampler, epoch = 1000, batch_size = 64, learning_rate = 0.01):
  # for _ in range(epoch):
  #   batch_loader = sampler.sample(batch_size = batch_size)
  #   delta_ws = np.zeros_like(rbm['weight'])
  #   delta_theta1s = np.zeros_like(rbm['theta1'])
  #   delta_theta2s = np.zeros_like(rbm['theta2'])

  #   # todo here is the thing
  #   for data in batch_loader:
  #     b0, raw_h, v = rbm.forward(data, iter = 20)
  #     # The key lies here yo
  #     # xxxx = np.outer(data, np.tanh(b0))
  #     # yyyy = np.outer(v, np.tanh(raw_h))

  #     # delta_ws.append(learning_rate*(np.tanh(np.dot(b0, data)) - np.tanh(np.dot(raw_h, v))))
  #     delta_ws += learning_rate * (np.outer(data, np.tanh(b0)) - np.outer(v, np.tanh(raw_h)))
  #     delta_theta1s += -learning_rate*(data - v)
  #     delta_theta2s += -learning_rate*(np.tanh(b0) - np.tanh(raw_h))
  #   rbm['weight'] += delta_ws
  #   rbm['theta1'] += delta_theta1s
  #   rbm['theta2'] += delta_theta2s

  for _ in range(epoch):
    batch_loader = sampler.sample(batch_size = batch_size)

    b0, raw_h, v = rbm.forward(batch_loader, iter = 10)
    delta_ws = learning_rate * (np.tensordot(np.tanh(b0), batch_loader, axes=([0], [0])) - 
                                np.tensordot(np.tanh(raw_h), v, axes=([0], [0])))
    delta_theta1s = -learning_rate * np.sum(batch_loader - v, axis=0)
    delta_theta2s = -learning_rate * np.sum(np.tanh(b0) - np.tanh(raw_h), axis=0)

    rbm['weight'] += delta_ws.T
    rbm['theta1'] += delta_theta1s
    rbm['theta2'] += delta_theta2s

  return rbm

if __name__ == "__main__":
  sampler = sampler()
  for i in [1,2,4,8]:
    nn = RBMachine(M=i)
    # batch_size = 32
    nn = train(nn, sampler=sampler)
    with open(f'cdk10_trained_rbm_model{i}.pkl', 'wb') as f:
      pickle.dump(nn, f)
    print(f'train done for {i}')