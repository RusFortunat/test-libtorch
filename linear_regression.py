import torch as T
import torch.nn as nn
import time

start_time = time.time() # time your simulation

# Hyper parameters
input_size = 1
output_size = 1
num_epochs = 60
learning_rate = 0.001

# Sample dataset
x_train = T.randn(15,1, requires_grad=True)
y_train = T.randn(15,1)

# Linear regression model
model = nn.Linear(input_size, output_size)

# Optimizer
optimizer = T.optim.SGD(model.parameters(), lr=learning_rate) # give network an optimizer


print("Training...\n")

# Train the model
for epoch in range(0, num_epochs):
    # Forward pass
    Loss = nn.MSELoss()
    output = model.forward(x_train)
    #print(output)
    loss = Loss(output, y_train)

    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if ((epoch + 1) % 5 == 0):
        print("Epoch [", (epoch + 1), "/", num_epochs, "], Loss: ", loss, "\n")
    


print("Training finished!\n")
print("Python execution time: %s seconds " % (time.time() - start_time))