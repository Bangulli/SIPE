import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.autograd import Function
from src.utils.misc import make_name_from_list

class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

# Classifier ######################################################## 

class Classif(nn.Module):
    def __init__(self, inputs, n_classes):
        super().__init__()
        self.norm = nn.ReLU()
        self.layer1 = nn.Linear(inputs, n_classes)
        self.act = nn.Softmax()
        
    def forward(self, x):
        x = self.norm(x)
        x = self.layer1(x)
        #x = self.act(x) training needs logits
        return x
    
class FullClassif(nn.Module):
    def __init__(self, stain_size, morph_size, possible_classes):
        super().__init__()
        self.stain_size = stain_size
        self.possible_classes = sorted(self.defrag(list(possible_classes.keys()))) ## for reproducibility
        self.enc = LabelEncoder()
        #print(len(self.possible_classes), self.possible_classes)
        self.enc.fit(self.possible_classes)
        #print(len(self.enc.classes_), self.enc.classes_)
        self.stain = Classif(stain_size, len(self.possible_classes))
        self.morph = Classif(morph_size, len(self.possible_classes))
        
    def forward(self, x):
        stain = self.stain(x[:, :self.stain_size])
        morph = self.morph(self.reverse_grad(x[:, self.stain_size:]))
        return stain, morph
    
    def transform_labels(self, labels):
        labels = self.defrag(labels)
        return self.enc.transform(labels)
    
    def reverse_grad(self, x, alpha=1.0):
        return GradientReversal.apply(x, alpha)
    
    def defrag(self, raw_labels):
        new_labels = []
        for label in raw_labels:
            label = make_name_from_list(label)
            ## hematox & eosin
            if label == "HE - Hematoxylin and eosin stain method (procedure)" or label == "Hematoxylin and eosin stain method" or label == "hematoxylin stain+water soluble eosin stain":
                new_labels.append('H&E')
            ## Van Gieson
            elif label == "Van Gieson stain" or label == "Verhoeff-Van Gieson stain method":
                new_labels.append("Van Gieson stain")
            elif "Periodic acid Schiff stain" in label and "blue" not in label:
                new_labels.append("Periodic acid Schiff stain")
            elif "Herovici's stain method"==label or "Herovic's stain method"==label: new_labels.append("Herovicis stain method")
            else: new_labels.append(label)
        return new_labels

# ConvClassifier ######################################################## 

class ConvClassif(nn.Module):
    def __init__(self, inputs, n_classes):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=16)
        self.norm = nn.ReLU()
        self.layer1 = nn.Linear(inputs, n_classes)
        self.act = nn.Softmax()
        
    def forward(self, x):
        x = self.pool(x)
        x = x.squeeze()
        #x = self.norm(x)
        x = self.layer1(x)
        #x = self.act(x) training needs logits
        return x
    
class FullConvClassif(nn.Module):
    def __init__(self, stain_size, morph_size, possible_classes):
        super().__init__()
        self.stain_size = stain_size
        self.possible_classes = sorted(self.defrag(list(possible_classes.keys()))) ## for reproducibility
        self.enc = LabelEncoder()
        #print(len(self.possible_classes), self.possible_classes)
        self.enc.fit(self.possible_classes)
        #print(len(self.enc.classes_), self.enc.classes_)
        self.stain = ConvClassif(stain_size, len(self.possible_classes))
        self.morph = ConvClassif(morph_size, len(self.possible_classes))
        
    def forward(self, x):
        stain = self.stain(x[:, :self.stain_size])
        morph = self.morph(self.reverse_grad(x[:, self.stain_size:]))
        return stain, morph
    
    def transform_labels(self, labels):
        labels = self.defrag(labels)
        return self.enc.transform(labels)
    
    def reverse_grad(self, x, alpha=1.0):
        return GradientReversal.apply(x, alpha)
    
    def defrag(self, raw_labels):
        new_labels = []
        for label in raw_labels:
            label = make_name_from_list(label)
            ## hematox & eosin
            if label == "HE - Hematoxylin and eosin stain method (procedure)" or label == "Hematoxylin and eosin stain method" or label == "hematoxylin stain+water soluble eosin stain":
                new_labels.append('H&E')
            ## Van Gieson
            elif label == "Van Gieson stain" or label == "Verhoeff-Van Gieson stain method":
                new_labels.append("Van Gieson stain")
            elif "Periodic acid Schiff stain" in label and "blue" not in label:
                new_labels.append("Periodic acid Schiff stain")
            elif "Herovici's stain method"==label or "Herovic's stain method"==label: new_labels.append("Herovicis stain method")
            else: new_labels.append(label)
        return new_labels