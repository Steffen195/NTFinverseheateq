import torch.nn as nn
import torch


class TemperatureNet(nn.Module):
    def __init__(self,hparam):
        super().__init__()
        self.hparam = hparam


        self.model = nn.Sequential()
        self.input_layer = nn.Sequential(nn.Linear(2, hparam["width"]),hparam["activation"])

        for i in range(hparam["num_layers_temperature"]):
            self.model.append(nn.Linear(hparam["width"], hparam["width"]))
            self.model.append(hparam["activation"])

        self.output_layer = nn.Linear(hparam["width"], 1)
                        

    def forward(self, x):
        x = self.input_layer(x)
        #skip1 = x
        x = self.model(x)
        #x = x + skip1
        x = self.output_layer(x)
        return x

class SourceNet(nn.Module):
    def __init__(self,hparam):
        super().__init__()
        self.hparam = hparam

        self.input_layer = nn.Sequential(nn.Linear(2, hparam["width"]),hparam["activation"])
        self.second_layer = nn.Sequential(nn.Linear(hparam["width"], hparam["width"]),hparam["activation"])
        
        self.model = nn.Sequential()
        for i in range(hparam["num_layers_source"]):
            self.model.append(nn.Linear(hparam["width"], hparam["width"]))
            self.model.append(hparam["activation"])

        self.last_hidden_layer = nn.Sequential(nn.Linear(hparam["width"], hparam["width"]),hparam["activation"])
        self.output_layer = nn.Linear(hparam["width"], 1)
        self.output_activation = hparam["output_activation"]

    def forward(self, x):
        x = self.input_layer(x)
        #skip1 = x
        x = self.second_layer(x)
        #skip2 = x
        x = self.model(x)
        
        #x = x + skip2
        x = self.last_hidden_layer(x)
        #x = x + skip1
        x = self.output_layer(x)
        x = self.output_activation(x)
        return x
    

class HeatNet(nn.Module):
    def __init__(self,hparam):
        super().__init__()
        self.hparam = hparam
        self.temperature_net = TemperatureNet(hparam)
        self.source_net = SourceNet(hparam)

    def forward(self, x):
        output = torch.concat((self.temperature_net(x), self.source_net(x)), dim = 1)
        return output
    

class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)
    

class AdaptiveSoftPlus(nn.Module):
    def __init__(self, beta = 1., threshold = 20.):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(threshold))
        self.threshold.requires_grad = True
        self.beta = nn.Parameter(torch.tensor(beta))
        self.beta.requires_grad = True
    def forward(self, x):
        return  1/self.beta * torch.log(1 + torch.exp(self.beta * x)) * (x < self.threshold) + x * (x >= self.threshold)
    

class AdaptiveTanh(nn.Module):
    def __init__(self,a = 1.):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(a))
        self.a.requires_grad = True
    def forward(self, x):
        return  torch.tanh(self.a)*torch.tanh(x)