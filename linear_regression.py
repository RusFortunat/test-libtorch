import torch as T
import torch.nn as nn
import time
import matplotlib.pyplot as plt

start_time = time.time() # time your simulation

# Hyper parameters
input_size = 1
output_size = 1
num_epochs = 60
learning_rate = 0.001
a=15
b=1
r1=0
r2=10

# Sample dataset
x_train = T.FloatTensor(a, b).uniform_(r1, r2) # creates random distribution in range 0:10
y_train = T.FloatTensor(a, b).uniform_(r1, r2)

# Linear regression model
model = nn.Linear(input_size, output_size)

# Optimizer
optimizer = T.optim.SGD(model.parameters(), lr=learning_rate)

print("Training...\n")

# Train the model
for epoch in range(0, num_epochs):
    # Forward pass
    output = model.forward(x_train)

    # Backward pass and optimize
    Loss = nn.MSELoss()
    loss = Loss(output, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if ((epoch + 1) % 5 == 0):
        print("Epoch [", (epoch + 1), "/", num_epochs, "], Loss: ", loss.item())
    
print("Training finished!\n")
print("Python execution time: %s seconds " % (time.time() - start_time))

output_filename = "output/linear_reggression.png"
plt.figure(figsize=(10,8))
axes = plt.gca()
plt.xlabel("X", fontsize=24)
plt.ylabel("Y", fontsize=24)
plt.xlim([r1,r2])
plt.ylim([r1,r2])
plt.tick_params(axis='both', which='major', direction = 'in', length = 10 , 
    width = 1, labelsize=24, bottom = True, top = True, left = True, right = True)
plt.scatter(x_train.detach().numpy(),y_train.detach().numpy(), c='k')
plt.plot(x_train.detach().numpy(),output.detach().numpy())
plt.tight_layout()
plt.savefig(output_filename, format="png", dpi=600)
plt.show()
