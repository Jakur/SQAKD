import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as T
from sklearn.preprocessing import StandardScaler

def train_linear(model, train_loader, test_loader, device, normalize=True):
    from sklearn.linear_model import LogisticRegression 
    normalize = T.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225])
    )
    train_set = []
    train_labels = []
    test_set = []
    test_labels = []
    model = model.eval()
    with torch.no_grad():
        for batch in train_loader:
            x, y, _ = batch
            x = x.to(device)
            if normalize:
                x = normalize(x)
            pred = model(x)
            train_set.append(pred.cpu())
            train_labels.append(y)

        for batch in test_loader:
            x, y, _ = batch
            x = x.to(device)
            if normalize:
                x = normalize(x)
            pred = model(x)
            test_set.append(pred.cpu())
            test_labels.append(y)

        train_set = torch.cat(train_set)
        train_labels = torch.cat(train_labels)
        test_set = torch.cat(test_set)
        test_labels = torch.cat(test_labels)
    train_set = np.array(train_set)
    test_set = np.array(test_set)
    scaler = StandardScaler()
    regr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=100)
    regr.fit(scaler.fit_transform(train_set), train_labels)
    mean = regr.score(scaler.transform(test_set), test_labels) # 25.5%, 26.2%
    return mean
    # print(f"Accuracy: {mean}")
    # import sklearn
    # import matplotlib.pyplot as plt
    # from sklearn.decomposition import PCA
    # from sklearn.manifold import TSNE 
    # pca = PCA(n_components=50)
    # test_transform = pca.fit_transform(test_set)
    # tsne = TSNE(2)
    # test_transform = tsne.fit_transform(test_transform)

    # f, ax = plt.subplots()
    # rand_cls = [1, 2, 17, 50, 99]
    # for cls in rand_cls:
    #     idx = test_labels == cls
    #     data = test_transform[idx]
    #     ax.scatter(data[:, 0], data[:, 1], label=cls)
    # f.legend()
    # f.savefig("fig.png", transparent=False)
