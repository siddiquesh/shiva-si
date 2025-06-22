from shiva.si_format import load_si
from torch import nn

# Define your model class here again
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

model, metadata = load_si(MyModel, "model.si")
print("✅ Loaded model with metadata:", metadata)

# You can now run evaluation or inference