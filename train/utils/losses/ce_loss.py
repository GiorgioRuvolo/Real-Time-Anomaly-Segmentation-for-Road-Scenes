import torch 

# Details: https://pytorch.org/docs/main/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss

class CrossEntropyLoss2d(torch.nn.Module):

    def __init__(self, weight=None):
        super().__init__()

        self.loss = torch.nn.NLLLoss2d(weight)
        # self.loss = torch.nn.NLLLoss(weight)

    def forward(self, outputs, targets):
        return self.loss(torch.nn.functional.log_softmax(outputs, dim=1), targets)
    
    def __str__(self):
        return "CrossEntropyLoss2d"