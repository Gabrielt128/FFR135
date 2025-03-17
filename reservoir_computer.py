import numpy as np

class reservoir_computer():
  def __init__(self) -> None:
    self.w1 = np.random.normal(scale=np.sqrt(0.002), size=(500,3))
    self.res_w = np.random.normal(scale=np.sqrt(0.004), size=(500,500))
    # self.reservoir_state = np.zeros((500,))
    self.reservoir_state = np.random.uniform(0, 1, size=(500,))

  def half_forward(self, X):
    # return the raw_output
    reservoir_states = []

    sum2 = X@(self.w1.T)
    for s in sum2:
      self.reservoir_state = np.tanh(
        s + self.res_w@self.reservoir_state
      )
      reservoir_states.append(self.reservoir_state)
    return reservoir_states
  
  def pred(self, r):
    return (r @ self.w2)  #(3,)

  def free_forward(self, times):
    # return the continuation
    current_o = self.pred(self.reservoir_state)
    continuation = []
    for _ in range(times):
      current_fake_x = current_o
      self.reservoir_state = np.tanh(current_fake_x@self.w1.T + self.res_w@self.reservoir_state)
      
      current_o = self.pred(self.reservoir_state)
      continuation.append(current_o)
    return continuation

def out_trainer(computer, raw):
    I = np.eye(raw.shape[1])
    w2 = np.linalg.inv(raw.T@raw + 0.01*I) @ raw.T @ train_data[1:,:]
    setattr(computer, 'w2', w2)


if __name__ == "__main__":
  train_data = np.loadtxt('training-set.csv', delimiter=',')
  test_data = np.loadtxt('test-set-8.csv', delimiter=',')

  train_data = train_data.T
  test_data = test_data.T

  computer = reservoir_computer()
  raw = np.array(computer.half_forward(train_data))
  print(raw.shape)

  out_trainer(computer, raw[:19899, :])
  print(computer.w2.shape)

  computer.reservoir_state = np.zeros((500,))
  _ = computer.half_forward(test_data)
  continuation = computer.free_forward(500)
  print(continuation)
  print(np.array(continuation).shape)

  continuation_array = np.array(continuation)
  y_continuation = continuation_array[:, 1]

  np.savetxt("prediction.csv", y_continuation, delimiter=",")