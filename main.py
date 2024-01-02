from torchcp.classification.scores import THR, APS, SAPS, RAPS
from torchcp.classification.predictors import SplitPredictor, ClusterPredictor, ClassWisePredictor
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import Subset
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

def transform_test_with_channel_check(image):
    channels = image.getbands()
    if len(channels) == 1:
        image = image.convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(size=224),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
    ])
    return transform(image)

def plot_results_dict():
    markers = ['o', 's', '^', 'D']
    plt.figure(figsize=(8,6))

    for i, score_func in enumerate(score_functions):
        plt.plot(alphas, results_dict[score_func]['coverage_rates'], label=f'{score_func} Coverage Rate', marker=markers[i])
    plt.xlabel('Alphas')
    plt.ylabel('Coverage Rate')
    plt.legend()
    plt.savefig('images/coverage_rate.png')
    plt.show()

    plt.figure(figsize=(8,6))
    for i, score_func in enumerate(score_functions):
        plt.plot(alphas, results_dict[score_func]['average_sizes'], label=f'{score_func} Average Size', marker=markers[i])
    plt.xlabel('Alphas')
    plt.ylabel('Average Size')
    plt.legend()
    plt.savefig('images/average_size.png')
    plt.show()

def plot_results_dict_fun_pred():
    plt.figure(figsize=(8, 6))
    markers = ['o', 's', '^', 'D']
    for i, score_func in enumerate(score_functions):
        plt.plot(predictors, [results_dict_fun_pred[score_func][predictor]['coverage_rate'] for predictor in predictors], label=f'{score_func} Coverage Rate', marker=markers[i])
    plt.xlabel('Predictors')
    plt.ylabel('Coverage Rate')
    plt.legend()
    plt.savefig('images/coverage_rate_predictors.png')
    plt.show()

    plt.figure(figsize=(8, 6))
    for i, score_func in enumerate(score_functions):
        plt.plot(predictors, [results_dict_fun_pred[score_func][predictor]['average_sizes'] for predictor in predictors], label=f'{score_func} Average Size', marker=markers[i])
    plt.xlabel('Predictors')
    plt.ylabel('Average Size')
    plt.legend()
    plt.savefig('images/average_size_predictors.png')
    plt.show()

def test_one():
    for alpha in tqdm(alphas, desc='Processing alphas', unit='alpha'):
        predictor = SplitPredictor(score_function, model)
        predictor.calibrate(cal_dataloader, alpha)
        examples = next(iter(test_dataloader))
        tmp_x, tmp_label = examples[0][0].unsqueeze(0), examples[1][0]
        with torch.no_grad():
            prediction_sets = predictor.predict(tmp_x)
            print("alpha: ", alpha)   
            true_class_name = class_names[tmp_label.item()]
            print("true_class_name: ", true_class_name)
            predicted_classes = [class_names[idx] for idx in prediction_sets[0]]
            print("predicted_classes: ", predicted_classes)
    examples = next(iter(test_dataloader))
    first_image = examples[0][0].permute(1, 2, 0).numpy()
    plt.imshow(first_image, cmap='gray')
    plt.savefig('images/first_image.png')
    plt.show()

if __name__ == '__main__':
    num_classes = 10
    fashion_mnist_test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_test_with_channel_check)
    class_names = fashion_mnist_test_dataset.classes

    # 划分测试集为cal和test
    total_test_size = len(fashion_mnist_test_dataset)
    cal_size = total_test_size // 2
    test_size = total_test_size - cal_size
    cal_indices, test_indices = torch.utils.data.random_split(range(total_test_size), [cal_size, test_size])
    cal_dataset = Subset(fashion_mnist_test_dataset, cal_indices)
    test_dataset = Subset(fashion_mnist_test_dataset, test_indices)
    batch_size = 128
    cal_dataloader = DataLoader(cal_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    #model_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.eval()
    #model = model.to(model_device)

    # score_functions = ["APS"]
    score_functions = ["THR", "SAPS", "RAPS", "APS"]
    results_dict = {score_func: {'coverage_rates': [], 'average_sizes': []} for score_func in score_functions}

    alphas = [0.99,0.95,0.90,0.75,0.5,0.25,0.1,0.05,0.01]
    for score_func in score_functions:
        for alpha in tqdm(alphas, desc=f'Processing alphas for {score_func}', unit='alpha'):
            if score_func == "THR":
                predictor = SplitPredictor(THR(), model)
            elif score_func == "APS":
                predictor = SplitPredictor(APS(), model)
            elif score_func == "SAPS":
                predictor = SplitPredictor(SAPS(1), model)
            elif score_func == "RAPS":
                predictor = SplitPredictor(RAPS(0.2), model)
            else:
                raise Exception("score_func error")
            predictor.calibrate(cal_dataloader, alpha)
            result = predictor.evaluate(test_dataloader)
            coverage_rate = result['Coverage_rate']
            average_size = result['Average_size']
            results_dict[score_func]['coverage_rates'].append(coverage_rate)
            results_dict[score_func]['average_sizes'].append(average_size)
    plot_results_dict()

    predictors = ["SplitPredictor", "ClusterPredictor", "ClassWisePredictor"]
    results_dict_fun_pred = {
        score_func: {predictor: {'coverage_rate': 0, 'average_sizes': 0} for predictor in predictors}
        for score_func in score_functions
    }
    alpha = 0.1
    for score_func in score_functions:
        for predictor_name in tqdm(predictors, desc=f'Processing alphas for {score_func}', unit='predictor'):
            if score_func == "THR":
                score_function  = THR()
            elif score_func == "APS":
                score_function  = APS()
            elif score_func == "SAPS":
                score_function  = SAPS(1)
            elif score_func == "RAPS":
                score_function  = RAPS(0.2)
            else:
                raise Exception("score_func error")

            if predictor_name == "SplitPredictor":
                predictor = SplitPredictor(score_function, model)
            elif predictor_name == "ClusterPredictor":
                predictor = ClusterPredictor(score_function, model)
            elif predictor_name == "ClassWisePredictor":
                predictor = ClassWisePredictor(score_function, model)
            else:
                raise Exception("predictor error")

            predictor.calibrate(cal_dataloader, alpha)
            result = predictor.evaluate(test_dataloader)
            coverage_rate = result['Coverage_rate']
            average_size = result['Average_size']
            results_dict_fun_pred[score_func][predictor_name]['coverage_rate'] = coverage_rate
            results_dict_fun_pred[score_func][predictor_name]['average_sizes'] = average_size
    plot_results_dict_fun_pred()
    test_one()
    print("results_dict = ", results_dict)
    print("results_dict_fun_pred = ", results_dict_fun_pred)
